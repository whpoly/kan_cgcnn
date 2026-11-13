"""Round two: inner-CV tuning with a global lock before outer-test evaluation."""
from __future__ import annotations

from run_modnet_official_matched import (
    SafePool, configure_official_runtime, digest, file_hash, jsonable,
    predict_members, replace_with_retry, sample_ids, set_seed, write_json)
import argparse
import contextlib
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
from modnet.preprocessing import MODData
from official_hybrid_models import HybridMODNetModel, PAIRS, VARIANTS, fit_hybrid

TASKS = ("matbench_steels", "matbench_phonons", "matbench_dielectric")
KAN_GRID = (3, 5, 8)
KAN_LR = (0.5, 1.0)
MLP_LR = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
STATUS_PATH = None
_TRAIN_CACHE = (None, None)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def now():
    return datetime.now(timezone.utc).isoformat()


def load_worker_training(path):
    global _TRAIN_CACHE
    if _TRAIN_CACHE[0] != path:
        _TRAIN_CACHE = (path, MODData.load(path))
    return _TRAIN_CACHE[1]


def evaluate_member(job):
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    try:
        identity = {k: v for k, v in job.items() if k != "directory"}
        signature = digest(identity)
        if (directory / "result.json").exists() and (directory / "model.pkl").exists():
            result = read(directory / "result.json")
            if result["signature"] != signature:
                raise ValueError(f"Incompatible cached member: {directory}")
            return result
        source = job["source"]
        outer_train = load_worker_training(job["training_data"])
        train, validation = outer_train.split((np.asarray(job["train_positions"]),
                                               np.asarray(job["validation_positions"])))
        if sample_ids(train) != source["train_ids"] or sample_ids(validation) != source["validation_ids"]:
            raise ValueError("Inner split differs from the paired MODNet member.")
        features = list(train.get_optimal_descriptors())[:source["genes"]["n_feat"]]
        if features != source["selected_features"]:
            raise ValueError("Input features differ from the paired MODNet member.")
        tf.keras.backend.clear_session()
        set_seed(source["seed"])
        started = time.perf_counter()
        write_json(directory / "status.json", dict(state="training", started_at=now()))

        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 25 == 0:
                    write_json(directory / "status.json", dict(state="training", epoch=epoch + 1,
                               elapsed_seconds=time.perf_counter() - started, logs=logs or {}))

        with (directory / "training.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model, validation_mae = fit_hybrid(
                    source["genes"], train, validation, variant=job["variant"],
                    grids=job["grids"], lr_multiplier=job["lr_multiplier"],
                    seed=source["seed"], smoke=job["smoke"], callbacks=[Progress()])
        if not np.isfinite(validation_mae):
            raise ValueError("Nonfinite inner validation score.")
        result = dict(
            signature=signature, run_fingerprint=job["run_fingerprint"], variant=job["variant"],
            base=job["base"], trial=job["trial"], inner=job["inner"], grids=job["grids"],
            lr_multiplier=job["lr_multiplier"], seed=source["seed"], genes=source["genes"],
            effective_lr=source["genes"]["lr"] * job["lr_multiplier"],
            source_member=source["directory"], selected_features=features,
            train_ids=sample_ids(train), validation_ids=sample_ids(validation),
            validation_mae=validation_mae, parameters=model.model.count_params(),
            target_parameters=model.target_parameters, parameter_error=model.parameter_error,
            mlp_hidden_width=model.mlp_hidden_width, num_neurons=model.num_neurons,
            epochs=len(model.history["loss"]), seconds=time.perf_counter() - started,
            smoke=job["smoke"], history=model.history, directory=str(directory))
        model._make_picklable()
        temporary = directory / "model.pkl.tmp"
        with temporary.open("wb") as handle:
            pickle.dump(model, handle, protocol=4)
        replace_with_retry(temporary, directory / "model.pkl")
        write_json(directory / "result.json", result)
        write_json(directory / "status.json", dict(state="complete", seconds=result["seconds"], finished_at=now()))
        del model, train, validation
        tf.keras.backend.clear_session()
        gc.collect()
        return result
    except Exception:
        write_json(directory / "status.json", dict(state="failed", traceback=traceback.format_exc()))
        raise


def choose_trials(records, expected_trials):
    """Only five-fold inner-validation MAE enters selection; stable ID breaks ties."""
    table, winners = [], []
    for base in sorted({r["base"] for r in records}):
        candidates = []
        base_records = [r for r in records if r["base"] == base]
        trials = sorted({r["trial"] for r in base_records})
        if len(trials) != expected_trials:
            raise ValueError("Incomplete candidate budget.")
        for trial in trials:
            members = sorted((r for r in base_records if r["trial"] == trial), key=lambda r: r["inner"])
            if [r["inner"] for r in members] != list(range(5)):
                raise ValueError("Selection requires every one of the five inner folds.")
            scores = [r["validation_mae"] for r in members]
            if not np.isfinite(scores).all():
                raise ValueError("Invalid validation scores.")
            item = dict(base=base, trial=trial, mean_validation_mae=float(np.mean(scores)),
                        validation_mae=scores, grids=members[0]["grids"],
                        lr_multiplier=members[0]["lr_multiplier"],
                        members=[dict(directory=r["directory"], seed=r["seed"], inner=r["inner"],
                                      parameters=r["parameters"], target_parameters=r["target_parameters"],
                                      source_member=r["source_member"]) for r in members])
            candidates.append(item)
            table.append(item)
        winners.append(min(candidates, key=lambda r: (r["mean_validation_mae"], r["trial"])))
    return winners, table


def recorded_inner_positions(outer_ids, source):
    """Recover the exact saved order, including FitGenetic's initial shuffle."""
    position = {sample: index for index, sample in enumerate(outer_ids)}
    train_ids, val_ids = source["train_ids"], source["validation_ids"]
    if (len(position) != len(outer_ids) or len(set(train_ids)) != len(train_ids)
            or len(set(val_ids)) != len(val_ids) or set(train_ids) & set(val_ids)
            or set(train_ids) | set(val_ids) != set(outer_ids)):
        raise ValueError("Invalid recorded inner partition.")
    return (np.asarray([position[s] for s in train_ids]),
            np.asarray([position[s] for s in val_ids]))


def prepare_fold(args, task, fold):
    source_dir = args.baseline_dir / task / f"fold-{fold}"
    output = args.output_dir / task / f"fold-{fold}"
    output.mkdir(parents=True, exist_ok=True)
    split = read(source_dir / "outer-split.json")
    data = MODData.load(source_dir / "official-train-moddata.pkl.gz")
    if sample_ids(data) != split["train_ids"]:
        raise ValueError("Outer training cache order changed.")
    member_names = [Path(m["directory"]).name for m in read(source_dir / "frozen-ensemble.json")["members"]]
    groups = {}
    for name in member_names:
        group, inner = name.rsplit("-inner-", 1)
        groups.setdefault(group, {})[int(inner)] = source_dir / "modnet" / name / "result.json"
    if len(groups) != 10 or any(set(group) != set(range(5)) for group in groups.values()):
        raise ValueError("Expected ten MODNet configurations, each with five inner members.")
    first_group = next(iter(groups.values()))
    inner_splits = [recorded_inner_positions(split["train_ids"], read(first_group[i])) for i in range(5)]
    if sorted(np.concatenate([val for _, val in inner_splits]).tolist()) != list(range(len(split["train_ids"]))):
        raise ValueError("Inner validation partitions do not cover the outer training set exactly once.")
    bases = []
    for base, group in enumerate(list(groups.values())[:1 if args.smoke else 10]):
        sources = []
        for inner in range(5):
            source = read(group[inner])
            train, val = data.split(inner_splits[inner])
            if sample_ids(train) != source["train_ids"] or sample_ids(val) != source["validation_ids"]:
                raise ValueError("Inner folds no longer match round one.")
            sources.append({key: source[key] for key in (
                "genes", "seed", "train_ids", "validation_ids", "selected_features", "parameters", "directory")})
        if any(s["genes"] != sources[0]["genes"] for s in sources):
            raise ValueError("A base configuration changes between inner folds.")
        bases.append(dict(base=base, sources=sources))
    inputs = dict(task=task, fold=fold, outer_split=split,
                  training_data=str(source_dir / "official-train-moddata.pkl.gz"),
                  training_data_sha256=file_hash(source_dir / "official-train-moddata.pkl.gz"),
                  ranked_features=list(data.get_optimal_descriptors()), bases=bases,
                  inner_splits=[dict(train_positions=a, validation_positions=b) for a, b in inner_splits])
    target = output / "inputs.json"
    if target.exists() and digest(read(target)) != digest(inputs):
        raise ValueError("Fold inputs changed.")
    write_json(target, inputs)
    return output, inputs


def make_jobs(args, output, inputs, variant, fingerprint, prior=None):
    jobs = []
    for base in inputs["bases"]:
        if variant.endswith("kan"):
            choices = [(g, lr) for g in ((3, 5) if args.smoke else KAN_GRID) for lr in KAN_LR]
        else:
            # Freeze the paired KAN's selected capacity before tuning the MLP.
            grids = next(w["grids"] for w in prior if w["base"] == base["base"])
            choices = [(grids, lr) for lr in ((0.5, 1.0, 1.5, 2.0) if args.smoke else MLP_LR)]
        for trial, (grids, lr) in enumerate(choices):
            for inner, source in enumerate(base["sources"]):
                jobs.append(dict(variant=variant, base=base["base"], trial=trial, inner=inner,
                                 grids=grids, lr_multiplier=lr, source=source,
                                 smoke=args.smoke, run_fingerprint=fingerprint,
                                 training_data=inputs["training_data"],
                                 training_data_sha256=inputs["training_data_sha256"],
                                 **inputs["inner_splits"][inner],
                                 directory=str(output / variant / f"base-{base['base']:02d}-trial-{trial:02d}-inner-{inner}")))
    return jobs


def tune_fold(args, task, fold, fingerprint, completed_folds):
    output, inputs = prepare_fold(args, task, fold)
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = read(selection_path)
        if selection["run_fingerprint"] != fingerprint:
            raise ValueError("Selection fingerprint mismatch.")
        return selection
    selections = {}
    with SafePool(args.n_jobs) as pool:
        for variant in VARIANTS:
            paired_kan = next((k for k, v in PAIRS.items() if v == variant), None)
            prior = selections[paired_kan]["winners"] if paired_kan else None
            jobs = make_jobs(args, output, inputs, variant, fingerprint, prior)
            records = []
            stage_start = time.perf_counter()
            for done, result in enumerate(pool.imap_unordered(evaluate_member, jobs), 1):
                records.append(result)
                print(f"{task} fold={fold} {variant}: {done}/{len(jobs)} "
                      f"base={result['base']} trial={result['trial']} inner={result['inner']} "
                      f"val_MAE={result['validation_mae']:.6g}", flush=True)
                write_json(STATUS_PATH, dict(state="running", pid=os.getpid(), phase="inner_search",
                           task=task, fold=fold, variant=variant, completed_members=done,
                           total_members=len(jobs), completed_search_folds=completed_folds,
                           expected_search_folds=len(args.tasks) * len(args.folds), smoke=args.smoke))
            winners, table = choose_trials(records, expected_trials=4 if args.smoke else 6)
            selections[variant] = dict(winners=winners, candidates=table,
                                       candidate_evaluations=len(records),
                                       search_member_seconds_sum=sum(r["seconds"] for r in records),
                                       phase_elapsed_seconds_this_invocation=time.perf_counter() - stage_start)
            write_json(output / f"{variant}-inner-search.json", selections[variant])
    for kan, mlp in PAIRS.items():
        for k, m in zip(selections[kan]["winners"], selections[mlp]["winners"]):
            for km, mm in zip(k["members"], m["members"]):
                if abs(mm["parameters"] / km["parameters"] - 1) > 0.01:
                    raise ValueError("Selected MLP is not within 1% of its paired KAN.")
    result = dict(task=task, fold=fold, run_fingerprint=fingerprint, smoke=args.smoke,
                  selection_rule="mean of five inner-validation MAEs per fixed base configuration",
                  selected_at=now(), outer_test_accessed=False, variants=selections)
    write_json(selection_path, result)
    return result


def evaluate_fold(args, task, fold, fingerprint, lock):
    output = args.output_dir / task / f"fold-{fold}"
    selection_path = output / "selection.json"
    key = str(selection_path.relative_to(args.output_dir))
    if file_hash(selection_path) != lock["selection_sha256"][key]:
        raise ValueError("Selection changed after the global lock.")
    if (output / "comparison.json").exists():
        result = read(output / "comparison.json")
        if result["run_fingerprint"] != fingerprint:
            raise ValueError("Output fingerprint mismatch.")
        return result
    selection, inputs = read(selection_path), read(output / "inputs.json")
    # This is the first stage that loads any outer-test data or labels.
    all_data = MODData.load(args.baseline_dir / "precomputed" / f"{task}_moddata.pkl.gz")
    split = inputs["outer_split"]
    _, test = all_data.split((np.asarray(split["train_positions"]), np.asarray(split["test_positions"])))
    if sample_ids(test) != split["test_ids"]:
        raise ValueError("Outer-test order differs from round one.")
    frame = pd.DataFrame(dict(sample_id=split["official_test_ids"], target=test.df_targets.iloc[:, 0].values))
    results = {}
    for variant in ("modnet", *VARIANTS):
        if variant == "modnet":
            jobs = [dict(directory=s["directory"], seed=s["seed"]) for b in inputs["bases"] for s in b["sources"]]
        else:
            jobs = [m for winner in selection["variants"][variant]["winners"] for m in winner["members"]]
        pred, std, replaced = predict_members(jobs, test, output, variant)
        members = [read(Path(j["directory"]) / "result.json") for j in jobs]
        frame[variant], frame[variant + "_std"] = pred, std
        results[variant] = dict(mae=float(np.mean(np.abs(frame.target.values - pred))),
                               ensemble_size=len(jobs), parameters_mean=float(np.mean([r["parameters"] for r in members])),
                               parameters_sum=sum(r["parameters"] for r in members),
                               selected_member_seconds_sum=sum(r["seconds"] for r in members),
                               search_member_seconds_sum=(selection["variants"][variant]["search_member_seconds_sum"]
                                                         if variant != "modnet" else None),
                               postprocess_replacements=replaced)
        frame.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    baseline = read(args.baseline_dir / task / f"fold-{fold}" / "baseline.json")
    if not args.smoke:
        np.testing.assert_allclose(results["modnet"]["mae"], baseline["results"]["modnet"]["mae"], atol=1e-9, rtol=0)
    result = dict(task=task, fold=fold, run_fingerprint=fingerprint, smoke=args.smoke,
                  selection_sha256=file_hash(selection_path), published_mae=baseline["published_modnet_mae"],
                  results=results)
    write_json(output / "comparison.json", result)
    return result


def manifest(args):
    baseline = read(args.baseline_dir / "matched-protocol.json")
    for path, expected in baseline["source_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Round-one dependency changed: {path}")
    packages = {name: importlib.metadata.version(name) for name in baseline["packages"]}
    if packages != baseline["packages"]:
        raise ValueError("Environment differs from the completed official reproduction.")
    sources = [Path(__file__), Path(__file__).with_name("official_hybrid_models.py"),
               Path(__file__).with_name("verify_modnet_hybrid_nested.py")]
    source_hashes = dict(baseline["source_sha256"], **{str(p.resolve()): file_hash(p) for p in sources})
    inputs = {}
    for task in args.tasks:
        paths = [args.baseline_dir / "precomputed" / f"{task}_moddata.pkl.gz"]
        for fold in args.folds:
            directory = args.baseline_dir / task / f"fold-{fold}"
            paths += [directory / name for name in ("official-train-moddata.pkl.gz", "outer-split.json", "frozen-ensemble.json")]
            paths += [Path(m["directory"]) / "result.json" for m in read(directory / "frozen-ensemble.json")["members"]]
        inputs.update({str(p): file_hash(p) for p in paths})
    return dict(kind="round_two_inner_tuned_hybrids", smoke=args.smoke, tasks=args.tasks, folds=args.folds,
                packages=packages, source_sha256=source_hashes, input_sha256=inputs,
                baseline_dir=str(args.baseline_dir), baseline_protocol_sha256=file_hash(args.baseline_dir / "matched-protocol.json"),
                variants=VARIANTS, training=baseline["training"], feature_selection=baseline.get("feature_selection",
                "Reuse round-one outer-training feature ranking in inner folds; not strictly nested feature selection."),
                parameter_tolerance=0.01, ensemble_members=5 if args.smoke else 50,
                kan_grid=[3, 5] if args.smoke else KAN_GRID, kan_lr_multipliers=KAN_LR,
                mlp_lr_multipliers=[0.5, 1.0, 1.5, 2.0] if args.smoke else MLP_LR,
                selection="Per fixed MODNet base configuration, mean five-fold validation MAE; no test-based tuning.",
                control="MLP capacity matches its paired KAN's inner-selected capacity; six LR trials per base.",
                gate="All requested outer-fold selections are frozen before any new outer-test evaluation.",
                hypotheses_predeclared=True,
                interpretation="Follow-up after observing round-one tests; exploratory benchmark, not a new untouched holdout.")


def main():
    global STATUS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir, args.baseline_dir = args.output_dir.resolve(), args.baseline_dir.resolve()
    STATUS_PATH = args.output_dir / "hybrid-status.json"
    if args.output_dir == args.baseline_dir or set(args.tasks) - set(TASKS) or set(args.folds) - set(range(5)):
        parser.error("Use a new directory and supported tasks/folds.")
    if not args.smoke and (args.tasks != list(TASKS) or args.folds != list(range(5))):
        parser.error("Production tuning locks all three tasks and all five outer folds together.")
    spec = manifest(args)
    fingerprint = digest(spec)
    protocol_path = args.output_dir / "hybrid-protocol.json"
    if protocol_path.exists() and digest(read(protocol_path)) != fingerprint:
        raise ValueError("Protocol/input/source changed; use a new output directory.")
    write_json(protocol_path, spec)
    for path, expected in spec["source_sha256"].items():
        destination = args.output_dir / "source-snapshot" / Path(path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)
        if file_hash(destination) != expected:
            raise ValueError("Snapshot mismatch.")
    completed = 0
    for task in args.tasks:
        for fold in args.folds:
            write_json(STATUS_PATH, dict(state="running", pid=os.getpid(), phase="prepare_inner_search",
                       task=task, fold=fold, completed_search_folds=completed, smoke=args.smoke))
            tune_fold(args, task, fold, fingerprint, completed)
            completed += 1
    lock = dict(run_fingerprint=fingerprint, scope="all_requested_folds", smoke=args.smoke,
                selection_sha256={str(p.relative_to(args.output_dir)): file_hash(p)
                                  for p in sorted(args.output_dir.glob("matbench_*/fold-*/selection.json"))})
    lock_path = args.output_dir / "selections-locked.json"
    if lock_path.exists() and read(lock_path) != lock:
        raise ValueError("Selection changed after outer-test lock.")
    write_json(lock_path, lock)
    for task in args.tasks:
        for fold in args.folds:
            write_json(STATUS_PATH, dict(state="running", pid=os.getpid(), phase="outer_evaluation",
                       task=task, fold=fold, smoke=args.smoke))
            evaluate_fold(args, task, fold, fingerprint, lock)
    from verify_modnet_hybrid_nested import verify
    verify(args.output_dir)
    write_json(STATUS_PATH, dict(state="complete", smoke=args.smoke, completed_at=now(),
                                completed_comparison_folds=len(args.tasks) * len(args.folds)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if STATUS_PATH is not None:
            write_json(STATUS_PATH, dict(state="failed", pid=os.getpid(), traceback=traceback.format_exc()))
        raise
