"""Refit the original ten KAN candidates per fold at 64 descriptors.

Use modnet-v012-matbench. Selection and outer scoring are separate; workers
receive only the outer-training cache and its five inner splits.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_phonons_input_budget import (
    OUTPUT as PRIOR, REFERENCE, native, read, spline, training_data,
)
from scaled_native_spline import fit_spline


OUTPUT = PRIOR.parent / "phonons-mm-kan64-reselect-20261114"
DENSE = PRIOR.parent / "phonons-mm-dense64-20261114"
N_FEATURES = 64


def model_dir(fold, name, inner):
    return OUTPUT / f"fold-{fold}" / name / f"inner-{inner}"


def fold_plan(previous, dense, reuse_audit):
    number = previous["fold"]
    folder = REFERENCE / "mm_kan" / f"fold-{number}"
    selection = folder / "frozen-selection.json"
    assert native.file_hash(selection) == previous["kan"]["selection_sha256"]
    saved = read(selection)
    assert saved["test_labels_used"] is False and len(saved["members"]) == 50
    ranking = sorted((score, gen, index) for gen in range(10) for index, score in
                     enumerate(read(folder / f"generation-{gen:02d}.json")["mean_validation_mae"]))
    assert len(ranking) == 200
    names = previous["ranking"][:N_FEATURES]
    splits = [dict(train_ids=m["train_ids"], val_ids=m["val_ids"]) for m in previous["members"]]
    assert dense["fold"] == number and dense["features"] == names and dense["splits"] == splits
    assert dense["official_test_ids"] == previous["official_test_ids"]
    audit = next(x for x in reuse_audit["fits"] if x["budget"] == N_FEATURES)
    configurations = []
    for rank, (score, gen, candidate) in enumerate(ranking[:10]):
        group = sorted((m for m in saved["members"] if
                        (m["generation"], m["candidate"]) == (gen, candidate)), key=lambda m: m["inner"])
        assert [m["inner"] for m in group] == list(range(5))
        genes = group[0]["genes"]
        widths, result_hashes = [], []
        for member, split in zip(group, splits):
            path = Path(member["directory"])
            fit = read(path / "result.json")
            assert not member["smoke"] and fit["variant"] == "bspline" and not fit["fast"]
            assert member["genes"] == fit["genes"] == genes
            assert member["train_ids"] == fit["train_ids"] == split["train_ids"]
            assert member["validation_ids"] == fit["validation_ids"] == split["val_ids"]
            assert fit["target_scaling"]["fit_ids"] == split["train_ids"]
            assert set(split["train_ids"] + split["val_ids"]).isdisjoint(previous["test_ids"])
            assert fit["selected_features"] == previous["ranking"][:genes["n_feat"]]
            assert native.file_hash(path / "model.pkl") == saved["model_sha256"][str(path)]
            widths.append(fit["spline_widths"])
            result_hashes.append(native.file_hash(path / "result.json"))
        assert all(w == widths[0] for w in widths) and widths[0][-1] == 1
        config = dict(name=f"kan-{rank:02d}", generation=gen, candidate=candidate,
                      original_mean_inner_mae=score, original_genes=genes,
                      hidden_widths=widths[0][:2], seeds=[m["seed"] for m in group],
                      original_result_sha256=result_hashes)
        if rank == 0:
            assert (gen, candidate) == (previous["best_generation"], previous["best_candidate"])
            assert all(m["widths"][:2] == config["hidden_widths"] for m in previous["members"])
            config["reuse"] = []
            for inner, sha in enumerate(audit["models_sha256"]):
                path = PRIOR / f"fold-{number}" / "k-064" / f"inner-{inner}"
                assert native.file_hash(path / "model.pkl") == sha
                config["reuse"].append(dict(directory=str(path), model_sha256=sha,
                                            result_sha256=native.file_hash(path / "result.json")))
        configurations.append(config)
    return dict(fold=number, features=names, cache=previous["cache"], cache_sha256=previous["cache_sha256"],
                train_ids=previous["train_ids"], test_ids=previous["test_ids"],
                official_test_ids=previous["official_test_ids"], splits=splits,
                original_kan_selection_sha256=native.file_hash(selection), configurations=configurations)


def freeze():
    previous = read(PRIOR / "protocol.json")
    dense = read(DENSE / "protocol.json")
    reuse = read(PRIOR / "model-audit.json")
    assert native.file_hash(PRIOR / "protocol.json") == dense["prior_protocol_sha256"]
    assert native.file_hash(Path(__file__).with_name("benchmark_phonons_input_budget.py")) == previous["trainer_sha256"]
    sources = [Path(__file__), Path(__file__).with_name("scaled_native_spline.py"),
               Path(__file__).with_name("native_fourway_models.py"),
               Path(__file__).with_name("run_modnet_official_matched.py")]
    plan = dict(task="matbench_phonons", bank="mm", features=N_FEATURES,
                candidate_configurations_per_fold=10, members_per_configuration=5,
                selection_rule="Lowest unweighted mean of five inner-validation MAEs at 64 features; "
                               "ties use the lower original rank. Selection is frozen before outer scoring.",
                training_rule="Retain each original candidate's two KAN hidden widths, spline grid, optimizer, "
                              "learning rate, input scaling, train-only target standardization, seeds and "
                              "train-loss early stopping. Change only input features to the same top 64 as MLP. "
                              "Maximum 800 epochs. No outer test samples in fitting or selection.",
                reuse_rule="Reuse the 25 already trained kan-00 members only after exact configuration, split, "
                           "seed, feature, parameter and saved-file hash checks. Train the remaining 225 members.",
                limitation="Candidate selection budgets are matched; KAN retains target standardization, "
                           "whereas the existing MLP experiment uses raw targets.",
                prior_protocol_sha256=native.file_hash(PRIOR / "protocol.json"),
                prior_fingerprint=native.digest(previous), prior_audit_sha256=native.file_hash(PRIOR / "model-audit.json"),
                dense_protocol_sha256=native.file_hash(DENSE / "protocol.json"),
                raw_data_sha256=previous["raw_data_sha256"],
                sources_sha256={str(p): native.file_hash(p) for p in sources},
                folds=[fold_plan(a, b, c) for a, b, c in zip(previous["folds"], dense["folds"], reuse)])
    path = OUTPUT / "protocol.json"
    if path.exists() and read(path) != plan:
        raise ValueError("The frozen KAN reselection protocol differs from this run")
    native.write_json(path, plan)
    return plan


def verify_fit(result, fold, config, inner, expected_fingerprint):
    split = fold["splits"][inner]
    assert result["run_fingerprint"] == expected_fingerprint
    assert result["variant"] == "bspline" and result["fast"] is False
    assert result["genes"] == dict(config["original_genes"], n_feat=N_FEATURES)
    assert result["seed"] == config["seeds"][inner]
    assert result["features"] == result["selected_features"] == fold["features"]
    assert result["train_ids"] == split["train_ids"] and result["validation_ids"] == split["val_ids"]
    assert set(result["train_ids"]).isdisjoint(result["validation_ids"])
    assert set(result["train_ids"] + result["validation_ids"]) == set(fold["train_ids"])
    assert set(result["train_ids"] + result["validation_ids"]).isdisjoint(fold["test_ids"])
    assert result["parameters"] == spline.spline_count(N_FEATURES, *config["hidden_widths"])
    assert native.np.isfinite(result["validation_mae"])
    assert 1 <= result["epochs"] <= 800


def fit(job):
    fold, config, inner, fingerprint, prior_fingerprint = job
    reused = "reuse" in config
    if reused:
        source = config["reuse"][inner]
        directory = Path(source["directory"])
        assert native.file_hash(directory / "model.pkl") == source["model_sha256"]
        assert native.file_hash(directory / "result.json") == source["result_sha256"]
        result = read(directory / "result.json")
    else:
        data, positions = training_data(fold["cache"], fold["cache_sha256"])
        names, split = fold["features"], fold["splits"][inner]
        assert names == list(data.get_optimal_descriptors())[:N_FEATURES] and len(names) == N_FEATURES
        assert set(positions) == set(fold["train_ids"])
        train, val = data.split(tuple([positions[sid] for sid in split[key]] for key in ("train_ids", "val_ids")))
        train.optimal_features, val.optimal_features = names, names
        targets = list(train.target_names)
        individual = SimpleNamespace(genes=dict(config["original_genes"], n_feat=N_FEATURES),
                                     targets=[[targets]], weights={t: 1 for t in targets}, num_classes=train.num_classes)
        directory = model_dir(fold["fold"], config["name"], inner)
        with patch.object(spline, "matched_widths", return_value=tuple(config["hidden_widths"])), \
             patch.object(native, "fit_kan", partial(fit_spline, seed=config["seeds"][inner])):
            result = native.evaluate_job(dict(individual=individual, train_data=train, val_data=val,
                        directory=str(directory), seed=config["seeds"][inner], fast=False,
                        variant="bspline", run_fingerprint=fingerprint))
    verify_fit(result, fold, config, inner, prior_fingerprint if reused else fingerprint)
    return dict(fold=fold["fold"], configuration=config["name"], inner=inner, directory=str(directory),
                reused=reused, epochs=result["epochs"], seconds=result["seconds"], parameters=result["parameters"],
                validation_mae=result["validation_mae"], model_sha256=native.file_hash(directory / "model.pkl"),
                result_sha256=native.file_hash(directory / "result.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    plan = freeze()
    fingerprint = native.digest(plan)
    jobs = [(f, f["configurations"][rank], inner, fingerprint, plan["prior_fingerprint"])
            for rank in range(10) for f in plan["folds"] for inner in range(5)]
    assert len(jobs) == 250
    if args.plan_only:
        for job in jobs[:25]:
            assert fit(job)["reused"]
        print("Frozen five folds x ten candidates x five members; 25 reusable members verified.", flush=True)
        return
    records = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit, job) for job in jobs]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                native.write_json(OUTPUT / "progress.json", dict(state="training", started_utc=started,
                    updated_utc=datetime.now(timezone.utc).isoformat(), completed=len(records), total=250,
                    reused=sum(x["reused"] for x in records), workers=args.workers, last=record))
                print(f"{len(records)}/250 fold={record['fold']} {record['configuration']} "
                      f"inner={record['inner']} epochs={record['epochs']} reused={record['reused']}", flush=True)
    except Exception as error:
        native.write_json(OUTPUT / "progress.json", dict(state="failed", completed=len(records), total=250,
                           error=repr(error), updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
    records.sort(key=lambda r: (r["fold"], r["configuration"], r["inner"]))
    assert len(records) == 250 and sum(x["reused"] for x in records) == 25
    native.write_json(OUTPUT / "fits.json", dict(protocol_fingerprint=fingerprint, records=records))
    native.write_json(OUTPUT / "progress.json", dict(state="complete", completed=250, total=250, reused=25,
                      newly_trained=225, started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
