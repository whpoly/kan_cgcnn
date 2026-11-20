"""Retrain pure Matminer phonons KANs at fixed numbers of input descriptors.

Use the modnet-v012-matbench environment. The separate scoring script reads
the outer targets after all fits. The full MODNet and KAN runs are read-only.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_modnet_official_matched as native
import native_fourway_models as spline
from native_fourway_features import OLD, ORIGINAL, ROOT, read
from scaled_native_spline import fit_spline


REFERENCE = ROOT / "benchmarks/native-fourway-scaled-kan-20261029"
OUTPUT = ROOT / "benchmarks/phonons-mm-input-budget-20261114"
BUDGETS = (16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224)


def reference_ensemble(arm, fold, ranking):
    folder = REFERENCE / arm / f"fold-{fold}"
    complete = read(folder / "complete.json")
    frozen = folder / "frozen-selection.json"
    assert native.file_hash(frozen) == complete["selection_sha256"]
    members = read(frozen)
    assert members["test_labels_used"] is False and len(members["members"]) == 50
    widths, columns = [], set()
    for member in members["members"]:
        fit = read(Path(member["directory"]) / "result.json")
        count = member["genes"]["n_feat"]
        assert fit["selected_features"] == ranking[:count] and len(fit["selected_features"]) == count
        widths.append(count)
        columns.update(fit["selected_features"])
    assert columns == set(ranking[:max(widths)])
    return dict(union=len(columns), member_widths=widths,
                selection_sha256=complete["selection_sha256"],
                prediction_sha256=complete["prediction_sha256"]), members["members"]


def fold_design(fold):
    feature_dir = REFERENCE / "inputs/mm" / f"fold-{fold}"
    manifest = read(feature_dir / "manifest.json")
    cache = feature_dir / "training.pkl.gz"
    assert native.file_hash(cache) == manifest["sha256"]
    split = read(OLD / "matbench_phonons" / f"fold-{fold}" / "outer-split.json")
    ranking = manifest["features"]
    assert len(set(ranking)) == len(ranking) and len(ranking) >= max(BUDGETS)
    assert split["train_ids"] == manifest["train_ids"] and split["test_ids"] == manifest["outer_test_ids"]
    train_ids, held = set(split["train_ids"]), set(split["test_ids"])
    assert (len(train_ids), len(held)) == (1012, 253) and train_ids.isdisjoint(held)
    baseline, _ = reference_ensemble("mm_modnet", fold, ranking)
    full_kan, members = reference_ensemble("mm_kan", fold, ranking)
    folder = REFERENCE / "mm_kan" / f"fold-{fold}"
    candidates = []
    for generation in range(10):
        record = read(folder / f"generation-{generation:02d}.json")
        native.np.testing.assert_allclose(native.np.mean(record["validation_mae"], axis=1),
                                          record["mean_validation_mae"], rtol=0, atol=1e-10)
        candidates.extend((loss, generation, i) for i, loss in enumerate(record["mean_validation_mae"]))
    _, generation, candidate = min(candidates)
    plan = []
    for index, member in enumerate(members):
        if (member["generation"], member["candidate"]) != (generation, candidate):
            continue
        fit = read(Path(member["directory"]) / "result.json")
        train, val = fit["train_ids"], fit["validation_ids"]
        assert set(train).isdisjoint(val) and set(train + val) == train_ids
        assert fit["target_scaling"]["fit_ids"] == train and fit["genes"] == member["genes"]
        plan.append(dict(inner=member["inner"], index_in_reference=index, seed=member["seed"],
                         train_ids=train, val_ids=val, genes=fit["genes"], widths=fit["spline_widths"]))
    plan.sort(key=lambda m: m["inner"])
    assert [m["inner"] for m in plan] == list(range(5))
    assert sorted(sid for m in plan for sid in m["val_ids"]) == sorted(train_ids)
    return dict(fold=fold, train_ids=split["train_ids"], test_ids=split["test_ids"],
                official_test_ids=split["official_test_ids"], cache=str(cache), cache_sha256=manifest["sha256"],
                ranking=ranking, modnet=baseline, kan=full_kan, best_generation=generation,
                best_candidate=candidate, members=plan)


def freeze_design():
    protocol = dict(task="matbench_phonons", bank="mm", budgets=list(BUDGETS),
                    relative_mae_margin=0.05, raw_data_sha256=native.file_hash(ORIGINAL),
                    method="For each outer fold, freeze the original KAN candidate with lowest mean inner "
                           "validation MAE. Use its five original inner splits, seeds, hidden widths, "
                           "scaling, optimizer and stopping rule. Retrain with exactly the first k columns "
                           "of that outer training fold's original MODNet relevance-redundancy ranking. "
                           "Equal-weight raw predictions. Compare to the original 50-member MODNet.",
                    trainer_sha256=native.file_hash(__file__), folds=[fold_design(f) for f in range(5)])
    ids = [sid for fold in protocol["folds"] for sid in fold["official_test_ids"]]
    assert len(ids) == len(set(ids)) == 1265
    path = OUTPUT / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("Different training design already exists in " + str(OUTPUT))
    native.write_json(path, protocol)
    return protocol


@lru_cache(maxsize=1)
def training_data(cache, sha):
    assert native.file_hash(cache) == sha
    data = native.MODData.load(cache)
    return data, {sid: i for i, sid in enumerate(data.df_targets.index)}


def member_dir(fold, budget, inner):
    return OUTPUT / f"fold-{fold}" / f"k-{budget:03d}" / f"inner-{inner}"


def fit_member(args):
    fold, budget, member, fingerprint = args
    data, positions = training_data(fold["cache"], fold["cache_sha256"])
    assert list(data.get_optimal_descriptors()) == fold["ranking"]
    train, val = data.split(tuple([positions[sid] for sid in member[key]] for key in ("train_ids", "val_ids")))
    names = fold["ranking"][:budget]
    train.optimal_features, val.optimal_features = names, names
    genes = dict(member["genes"], n_feat=budget)
    targets = list(train.target_names)
    individual = SimpleNamespace(genes=genes, targets=[[targets]], weights={t: 1 for t in targets},
                                 num_classes=train.num_classes)
    directory = member_dir(fold["fold"], budget, member["inner"])
    job = dict(individual=individual, train_data=train, val_data=val, directory=str(directory),
               seed=member["seed"], fast=False, variant="bspline", run_fingerprint=fingerprint)
    # Keep the original hidden widths fixed when changing the input dimension.
    with patch.object(spline, "matched_widths", return_value=tuple(member["widths"][:2])), \
         patch.object(native, "fit_kan", partial(fit_spline, seed=member["seed"])):
        result = native.evaluate_job(job)
    assert result["selected_features"] == names and result["train_ids"] == member["train_ids"]
    assert result["validation_ids"] == member["val_ids"]
    assert result["parameters"] == spline.spline_count(budget, *member["widths"][:2])
    return (fold["fold"], budget, member["inner"], result["epochs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    protocol = freeze_design()
    fingerprint = native.digest(protocol)
    jobs = [(fold, budget, member, fingerprint) for fold in protocol["folds"]
            for budget in BUDGETS for member in fold["members"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (fold, budget, inner, epochs) in enumerate(pool.map(fit_member, jobs), 1):
            print(f"{done}/{len(jobs)} fold={fold} k={budget} inner={inner} epochs={epochs}", flush=True)


if __name__ == "__main__":
    main()
