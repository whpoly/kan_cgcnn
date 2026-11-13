"""Paired B-spline KAN ablation of the frozen steel SHAP top-50 ensembles."""
import msvcrt
import traceback
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_steel396_shap_pair as pair
from scaled_native_spline import fit_spline
from native_fourway_models import matched_widths

nm, np = pair.nm, pair.np
REFERENCE = pair.BASE / "formal"
OUT = pair.ROOT / "benchmarks/steel396-shap-top50-kan-20261112"


def fit_member(job):
    with patch.object(nm, "fit_kan", partial(fit_spline, seed=job["seed"])):
        result = nm.evaluate_job(job)
    return result["directory"]


def prepare(data, y, splits):
    """Check all memberships, splits, and input prefixes before training."""
    plans = []
    for fold, (tr, te) in enumerate(splits):
        ref = REFERENCE / f"fold-{fold}"
        audit = pair.read(ref / "audit.json")
        selection = pair.read(ref / "top50/selection.json")
        assert audit["passed"] and audit["fp"] == selection["fp"]
        assert selection["members"] == pair.read(ref / "frozen.json")["members"]["top50"]
        train = pair.make_data(data.iloc[tr], y[tr])
        pairs = pair.fresh_pairs(train, 5)
        for inner, (it, val) in enumerate(pairs):
            rank = pair.read(ref / f"inner-{inner}/shap.json")
            assert rank["fp"] == selection["fp"] and rank["columns"] == list(data.columns)
            top = rank["subsets"]["top50"]
            assert top == list(data.columns[pair.selections(rank["mean_abs_shap"])["top50"]])
            assert rank["train_ids"] == nm.sample_ids(it)
            assert set(nm.sample_ids(it) + nm.sample_ids(val)) == set(data.index[tr])
            it.optimal_features, val.optimal_features = list(top), list(top)
        members = []
        for source in selection["members"]:
            directory = Path(source["directory"])
            fit = pair.read(directory / "result.json")
            it, val = pairs[source["inner"]]
            features = list(it.optimal_features)
            genes = fit["genes"]
            assert fit["variant"] == "modnet" and not fit["fast"] and fit["run_fingerprint"] == audit["fp"]
            assert fit["seed"] == source["seed"] and fit["features"] == features
            assert fit["selected_features"] == features[:genes["n_feat"]]
            assert fit["train_ids"] == nm.sample_ids(it) and fit["validation_ids"] == nm.sample_ids(val)
            matched_widths(genes["n_feat"], tuple(layer[0] for layer in nm.neurons_from_genes(genes)))
            members.append(dict(inner=source["inner"], genes=genes, seed=source["seed"],
                                features=features, parameters=fit["parameters"]))
        test_hash = pair.read(ref / "complete.json")["results"]["top50"]["sha256"]
        assert nm.file_hash(ref / "top50/test.npz") == test_hash
        assert len(members) == 50 and {m["inner"] for m in members} == set(range(5))
        plans.append((pairs, members, test_hash))
    return plans


def run():
    audit = pair.read(REFERENCE / "audit.json")
    assert audit["passed"] and not audit["smoke"]
    manifest = pair.read(pair.INPUT / "manifest.json")
    archive = pair.INPUT / manifest["archive"]
    assert nm.file_hash(archive) == audit["input_sha256"] == manifest["archive_sha256"]
    with np.load(archive, allow_pickle=False) as z:
        data = pair.pd.DataFrame(z["X"], index=z["sample_ids"], columns=z["columns"])
        y = z["y"]
        splits = [(z[f"train_positions_{i}"], z[f"test_positions_{i}"]) for i in range(5)]
    assert data.shape == (312, 396) and sorted(np.concatenate([te for _, te in splits])) == list(range(len(data)))
    plans = prepare(data, y, splits)
    scripts = Path(__file__).resolve().parent
    files = [Path(__file__).name, "scaled_native_spline.py", "native_fourway_models.py",
             "standalone_kan_models.py", "run_modnet_official_matched.py", "run_steel396_shap_pair.py",
             "run_native_fourway.py", "official_kan_models.py", "run_mattervial_phonons.py"]
    protocol = dict(reference=str(REFERENCE), input_sha256=audit["input_sha256"],
        method="Paired baseline-selected member settings and seeds; fresh parameter-matched B-spline KAN; inner-training-only target scaling; no new search.",
        folds=[dict(test_sha256=h, member_plan_sha256=nm.digest(m)) for _, m, h in plans],
        sources={f:nm.file_hash(scripts/f) for f in files}, workers=3)
    fp = nm.digest(protocol)
    if (OUT / "protocol.json").exists():
        assert pair.read(OUT / "protocol.json") == protocol
    nm.write_json(OUT / "protocol.json", protocol)
    rows = []
    with nm.SafePool(protocol["workers"]) as pool:
        for fold, ((_, te), (pairs, members, _)) in enumerate(zip(splits, plans)):
            folder = OUT / f"fold-{fold}"
            jobs = []
            for i, member in enumerate(members):
                it, val = pairs[member["inner"]]
                names = list(it.target_names)
                individual = SimpleNamespace(genes=member["genes"], targets=[[names]],
                    num_classes=it.num_classes, weights={n:1 for n in names})
                jobs.append(dict(directory=str(folder / f"member-{i:02d}"), seed=member["seed"],
                    run_fingerprint=fp, individual=individual, fast=False, variant="bspline",
                    train_data=it, val_data=val))
            if not (folder / "audit.json").exists():
                pair.status(OUT, "training", fold=fold, completed=0, total=len(jobs))
                for completed, _ in enumerate(pool.imap_unordered(fit_member, jobs), 1):
                    pair.status(OUT, "training", fold=fold, completed=completed, total=len(jobs))
                for member, job in zip(members, jobs):
                    r = pair.read(Path(job["directory"]) / "result.json")
                    assert r["run_fingerprint"] == fp and r["variant"] == "bspline"
                    assert r["genes"] == member["genes"] and r["seed"] == member["seed"]
                    assert r["features"] == member["features"] and r["selected_features"] == r["features"][:r["genes"]["n_feat"]]
                    assert abs(r["parameters"] / member["parameters"] - 1) <= .05
                pair.status(OUT, "outer_prediction", fold=fold, completed=len(jobs))
                pred, _, _ = nm.predict_members(jobs, pair.make_data(data.iloc[te], y[te]), folder, "outer")
                np.savez_compressed(folder/"test.npz", ids=data.index.to_numpy(dtype=str)[te], target=y[te], prediction=pred)
            with np.load(REFERENCE/f"fold-{fold}/top50/test.npz") as ref, np.load(folder/"test.npz") as kan, np.load(folder/"outer-member-predictions.npz") as ensemble:
                for z in (kan, ref):
                    np.testing.assert_array_equal(z["ids"], data.index[te])
                    np.testing.assert_array_equal(z["target"], y[te])
                    assert z["prediction"].shape == y[te].shape and np.isfinite(z["prediction"]).all()
                assert ensemble["predictions"].shape == (len(members), len(te))
                np.testing.assert_array_equal(kan["prediction"], ensemble["predictions"].mean(axis=0))
                baseline, model = [float(np.mean(np.abs(z["prediction"]-y[te]))) for z in (ref, kan)]
            row = dict(fold=fold, modnet=baseline, kan=model, delta_mpa=model-baseline)
            record = dict(passed=True, fp=fp, members=len(jobs), test_sha256=nm.file_hash(folder/"test.npz"), **row)
            if (folder / "audit.json").exists():
                assert pair.read(folder / "audit.json") == record
            nm.write_json(folder/"audit.json", record)
            rows.append(row)
    mean_kan, mean_modnet = [float(np.mean([r[k] for r in rows])) for k in ("kan", "modnet")]
    np.testing.assert_allclose(mean_modnet, pair.read(REFERENCE/"comparison.json")["top50"]["mean_mae_mpa"], rtol=0, atol=1e-8)
    nm.write_json(OUT/"comparison.json", dict(passed=True, folds=rows, mean_kan_mpa=mean_kan,
        mean_modnet_mpa=mean_modnet, delta_mpa=mean_kan-mean_modnet, delta_paper_mpa=mean_kan-pair.PAPER))
    pair.status(OUT, "complete", fits=sum(len(m) for _, m, _ in plans))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/"run.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            pair.status(OUT, "checking_frozen_inputs")
            run()
        except BaseException:
            pair.status(OUT, "failed", traceback=traceback.format_exc())
            raise
