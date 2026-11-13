"""Two prespecified steel MODNet experiments on identical folds and SHAP ranks."""
import argparse
import importlib.metadata as md
import json
import msvcrt
from pathlib import Path
import pickle
import traceback
from datetime import datetime, timezone

import run_modnet_official_matched as nm
from run_mattervial_phonons import make_data
from steels_paper_shap_kernel import importance

np, pd = nm.np, nm.pd
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmarks/steels-paper-reproduction-20261102/mm-roost396"
BASE = ROOT / "benchmarks/steel396-shap-pair-20261108"
RULES = ("mass50", "top50")
PAPER = 85.12
FIXED50 = 98.4658917595726
WORKERS = 3


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def status(out, state, **fields):
    nm.write_json(out / "status.json", dict(state=state, pid=nm.os.getpid(),
                  updated_at=datetime.now(timezone.utc).isoformat(), **fields))


def selections(values):
    weights = np.array(values, dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid mean absolute SHAP scores")
    order = np.argsort(-weights, kind="stable")
    count = int(np.searchsorted(np.cumsum(weights[order]), weights.sum()/2, side="left")) + 1
    return {"mass50": order[:count], "top50": order[:50]}


class SteelGA(nm.CheckpointGA):
    def function_fitness(self, pop, *args, **kwargs):
        # Upstream constructor requires at least two descriptors. If an inner
        # fold reaches half the SHAP mass with one, constrain the gene to one.
        for individual in pop:
            individual.genes["n_feat"] = min(individual.genes["n_feat"], self.width)
        return super().function_fitness(pop, *args, **kwargs)


def fresh_pairs(train, nested):
    return [train.split(split) for split in nm.matbench_kfold_splits(train, n_splits=nested)]


def stage_search(out, folder, fold, name, train, pairs, features, fp, seed, smoke):
    stage = folder / name
    record = stage / "selection.json"
    if record.exists():
        selected = read(record)
        assert selected["fp"] == fp
        return selected["members"]
    for (inner_train, val), cols in zip(pairs, features):
        inner_train.optimal_features = list(cols)
        val.optimal_features = list(cols)
    nm.set_seed(seed)
    ga = SteelGA(train, stage, seed, fp)
    ga.width = min(map(len, features))
    ga.train_data.optimal_features = list(train.df_featurized)[:max(2, ga.width)]
    ga.inner_data[len(pairs)] = pairs
    nm.set_seed(seed + 777)
    status(out, "genetic_search", fold=fold, stage=name, progress=str(stage / "progress.json"))
    try:
        ensemble = ga.run(n_jobs=WORKERS, fast=smoke, size_pop=50, num_generations=20,
                          nested=len(pairs), early_stopping=8, refit=False)
    except BaseException:
        if getattr(ga, "pool", None):
            ga.pool.terminate()
        raise
    members = []
    for model in ensemble.model:
        job = ga.job_by_path[model.path]
        p = Path(job["directory"])
        fit = read(p / "result.json")
        inner = int(p.name.rsplit("-inner-", 1)[1])
        assert fit["features"] == features[inner]
        assert nm.sample_ids(pairs[inner][0]) == fit["train_ids"]
        assert nm.sample_ids(pairs[inner][1]) == fit["validation_ids"]
        members.append(dict(directory=str(p), inner=inner, seed=job["seed"],
                            sha256=nm.file_hash(model.path), validation_mae=fit["validation_mae"]))
    nm.write_json(record, dict(fp=fp, members=members))
    return members


def get_ranking(folder, k, inner_train, members, fp, seed, smoke):
    path = folder / f"inner-{k}" / "shap.json"
    member = min((m for m in members if m["inner"] == k), key=lambda m: m["validation_mae"])
    if path.exists():
        previous = read(path)
        assert previous["fp"] == fp and previous["pilot"] == member
        return previous
    pilot = Path(member["directory"]) / "model.pkl"
    assert nm.file_hash(pilot) == member["sha256"]
    with pilot.open("rb") as stream:
        model = pickle.load(stream)
    model._restore_model()
    used = list(model.optimal_descriptors[:model.n_feat])
    columns = list(inner_train.df_featurized)
    x = inner_train.df_featurized[used].to_numpy()
    if model._scaler is not None:
        x = np.nan_to_num(model._scaler.transform(x), nan=-1)

    def predict(z):
        return np.concatenate([np.asarray(model.model(z[i:i+128], training=False)).reshape(-1)
                               for i in range(0, len(z), 128)])

    scores, rows, settings = importance(x, predict, seed, smoke)
    by_feature = dict(zip(used, scores))
    weights = [float(by_feature.get(c, 0.0)) for c in columns]
    subsets = {r: [columns[i] for i in ix] for r, ix in selections(weights).items()}
    ids = nm.sample_ids(inner_train)
    ranking = dict(fp=fp, pilot=member, train_ids=ids, background_ids=[ids[i] for i in rows],
                   columns=columns, mean_abs_shap=weights, settings=settings, subsets=subsets)
    nm.write_json(path, ranking)
    nm.tf.keras.backend.clear_session()
    return ranking


def run_fold(out, fold, data, targets, train_pos, test_pos, fp, smoke):
    folder = out / f"fold-{fold}"
    seed = 260902 + fold * 100000
    train = make_data(data.iloc[train_pos], targets[train_pos])
    train.optimal_features = list(data)
    pairs = fresh_pairs(train, 2 if smoke else 5)
    native = []
    for k, (it, _) in enumerate(pairs):
        status(out, "native_feature_ranking", fold=fold, inner=k)
        path = folder / f"inner-{k}" / "native.json"
        if not path.exists():
            nm.set_seed(seed + k)
            it.feature_selection(n=len(data.columns), n_jobs=1)
            nm.write_json(path, dict(fp=fp, features=list(it.optimal_features), train_ids=nm.sample_ids(it)))
        ranking = read(path)
        assert ranking["fp"] == fp and ranking["train_ids"] == nm.sample_ids(it)
        native.append(ranking["features"])
    initial = stage_search(out, folder, fold, "native", train, pairs, native, fp, seed, smoke)
    ranks = []
    for k, (it, _) in enumerate(pairs):
        status(out, "shared_shap_ranking", fold=fold, inner=k)
        ranks.append(get_ranking(folder, k, it, initial, fp, seed+k+524, smoke))
    frozen = {}
    for rule in RULES:
        features = [rank["subsets"][rule] for rank in ranks]
        frozen[rule] = stage_search(out, folder, fold, rule, train, pairs, features, fp, seed, smoke)
    # Both ensembles are fixed before either outer test is evaluated.
    nm.write_json(folder / "frozen.json", dict(fp=fp, members=frozen))
    test = make_data(data.iloc[test_pos], targets[test_pos])
    results = {}
    for rule in RULES:
        stage = folder / rule
        members = frozen[rule]
        for m in members:
            assert nm.file_hash(Path(m["directory"]) / "model.pkl") == m["sha256"]
        pred, _, replacements = nm.predict_members(members, test, stage, "outer")
        np.savez_compressed(stage / "test.npz", ids=data.index.to_numpy(dtype=str)[test_pos],
                            target=targets[test_pos], prediction=pred)
        results[rule] = dict(mae=float(np.abs(targets[test_pos]-pred).mean()),
                             replacements=int(replacements), sha256=nm.file_hash(stage/"test.npz"))
    nm.write_json(folder / "complete.json", dict(fp=fp, fold=fold, results=results))


def verify_fold(out, fold, ids, y, tr, te, fp, smoke):
    folder = out / f"fold-{fold}"
    result = read(folder / "complete.json")
    frozen = read(folder / "frozen.json")
    assert result["fp"] == frozen["fp"] == fp and result["fold"] == fold
    external = set(ids[te])
    outer_train = set(ids[tr])
    assert outer_train.isdisjoint(external) and len(outer_train | external) == len(ids)
    covered = set()
    for k in range(2 if smoke else 5):
        rank = read(folder / f"inner-{k}/shap.json")
        it = set(rank["train_ids"])
        val = outer_train - it
        assert it < outer_train and set(rank["background_ids"]) <= it
        assert val and val.isdisjoint(covered)
        covered.update(val)
        pilot = Path(rank["pilot"]["directory"])
        assert nm.file_hash(pilot/"model.pkl") == rank["pilot"]["sha256"]
        initial = read(pilot/"result.json")
        assert set(initial["train_ids"]) == it and set(initial["validation_ids"]) == val
        assert rank["fp"] == fp and rank["settings"]["max_additivity_residual"] < 1e-3
        scores = dict(zip(rank["columns"],rank["mean_abs_shap"]))
        ordered = sorted(scores, key=scores.get, reverse=True)
        top = rank["subsets"]["top50"]
        mass = rank["subsets"]["mass50"]
        assert top == ordered[:50] and len(top) == 50
        assert mass == ordered[:len(mass)] and len(mass) > 0
        half = sum(scores.values()) * .5
        assert sum(scores[c] for c in mass) >= half - 1e-12
        assert sum(scores[c] for c in mass[:-1]) < half + 1e-12
        for rule in RULES:
            jobs = [m for m in frozen["members"][rule] if m["inner"] == k]
            assert jobs
            for m in jobs:
                d = Path(m["directory"])
                fit = read(d/"result.json")
                assert set(fit["train_ids"]) == it and set(fit["validation_ids"]) == val
                assert fit["features"] == rank["subsets"][rule]
                assert nm.file_hash(d/"model.pkl") == m["sha256"]
    assert covered == outer_train
    for rule in RULES:
        stage = folder / rule
        assert read(stage/"selection.json")["members"] == frozen["members"][rule]
        assert nm.file_hash(stage/"test.npz") == result["results"][rule]["sha256"]
        with np.load(stage/"test.npz") as z, np.load(stage/"outer-member-predictions.npz") as ensemble:
            np.testing.assert_array_equal(z["ids"], ids[te])
            np.testing.assert_array_equal(z["target"], y[te])
            np.testing.assert_array_equal(z["prediction"], ensemble["predictions"].mean(axis=0))
            assert abs(np.abs(z["target"]-z["prediction"]).mean()-result["results"][rule]["mae"]) < 1e-8
    nm.write_json(folder/"audit.json", dict(passed=True, fp=fp, fold=fold, n_test=len(te)))
    return result["results"]


def source_hashes():
    names = [Path(__file__).name, "steels_paper_shap_kernel.py", "run_modnet_official_matched.py",
             "run_mattervial_phonons.py", "modnet_runtime.py", "prepare_steels_roost396.py"]
    return {n: nm.file_hash(ROOT/"scripts"/n) for n in names}


def execute(out, smoke):
    manifest = read(INPUT/"manifest.json")
    archive = INPUT/manifest["archive"]
    assert nm.file_hash(archive) == manifest["archive_sha256"]
    assert md.version("modnet") == "0.1.13"
    sources = source_hashes()
    protocol = dict(smoke=smoke, sources=sources, input_sha256=manifest["archive_sha256"],
        modnet=md.version("modnet"), groups=manifest["groups"], rules=RULES,
        assumptions="Both rules are our hypotheses, not confirmed author definitions. Each inner fold trains a fresh native GA. Its best validation member within the native ensemble supplies KernelSHAP on inner training rows. Columns unused by this member have zero attribution. Equal scores retain archive column order.",
        mass50="Smallest prefix reaching 50% of total mean absolute SHAP; no minimum number of features.",
        top50="Exactly the first 50 candidate features. MODNet may use a shorter prefix within either candidate bank.",
        training="Five official outer folds, 5 inner folds, three workers, population 50, at most 20 generations, patience 8, refit=False. Initial seed 260902 + outer_fold * 100000; paired inner splits and GA seeds.",
        shap="KernelExplainer 0.44.1, up to 300 inner training rows, nsamples=500, l1_reg=0. Inner validation participates in pilot selection. Outer test is held out until both arms are frozen.",
        reference_mpa={"paper": PAPER, "author_fixed50": FIXED50})
    fp = nm.digest(protocol)
    if (out/"protocol.json").exists():
        assert read(out/"protocol.json") == json.loads(json.dumps(protocol))
    nm.write_json(out/"protocol.json",protocol)
    if not smoke:
        gate = read(BASE/"smoke/audit.json")
        assert gate["passed"] and gate["sources"] == sources and gate["input_sha256"] == protocol["input_sha256"]
    with np.load(archive, allow_pickle=False) as z:
        ids, y = z["sample_ids"], z["y"]
        assert len(set(ids)) == len(ids) == 312
        assert z["X"].shape == (312,396) and list(z["columns"]) == manifest["columns"]
        data = pd.DataFrame(z["X"], index=ids, columns=z["columns"])
        if smoke:
            data = data.iloc[:, np.linspace(0,395,64,dtype=int)]
        splits = [(z[f"train_positions_{i}"], z[f"test_positions_{i}"]) for i in range(5)]
    assert sorted(np.concatenate([te for _,te in splits])) == list(range(len(ids)))
    rows = []
    for fold,(tr,te) in enumerate(splits[:1] if smoke else splits):
        if not (out/f"fold-{fold}/complete.json").exists():
            run_fold(out, fold, data, y, tr, te, fp, smoke)
        rows.append(verify_fold(out, fold, ids, y, tr, te, fp, smoke))
        # Keep selected models for verification, and all fit metadata. Nonselected
        # models are disposable only after this newly created fold passes audit.
        for name in ("native", *RULES):
            stage = out/f"fold-{fold}"/name
            keep = {Path(m["directory"]) for m in read(stage/"selection.json")["members"]}
            for path in (stage/"modnet").glob("generation-*-candidate-*-inner-*/model.pkl"):
                if path.parent not in keep and stage.resolve() in path.resolve().parents:
                    path.unlink()
    if not smoke:
        summary = {}
        for rule in RULES:
            maes = [r[rule]["mae"] for r in rows]
            mean = sum(maes)/len(maes)
            summary[rule] = dict(folds_mpa=maes, mean_mae_mpa=mean,
                                 delta_paper_mpa=mean-PAPER, delta_author_fixed50_mpa=mean-FIXED50)
        nm.write_json(out/"comparison.json",summary)
    nm.write_json(out/"audit.json",dict(passed=True,smoke=smoke,sources=sources,input_sha256=protocol["input_sha256"]))
    status(out,"complete")


def main(mode):
    out = BASE/mode
    out.mkdir(parents=True, exist_ok=True)
    with (out/"run.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            status(out, "starting")
            execute(out, mode == "smoke")
        except BaseException:
            status(out,"failed",traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["smoke","formal","sequence"])
    mode = parser.parse_args().mode
    for phase in (["smoke","formal"] if mode == "sequence" else [mode]):
        main(phase)
