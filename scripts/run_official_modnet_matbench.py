from __future__ import annotations

import argparse
import csv
import gzip
import inspect
import json
import os
import pickle
import time
from pathlib import Path
from traceback import print_exc
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

MATBENCH_TASKS = [
    "matbench_dielectric",
    "matbench_elastic",
    "matbench_expt_gap",
    "matbench_glass",
    "matbench_jdft2d",
    "matbench_mp_e_form",
    "matbench_mp_gap",
    "matbench_mp_is_metal",
    "matbench_perovskites",
    "matbench_phonons",
    "matbench_steels",
]
SMALL_MATBENCH_TASKS = [
    "matbench_dielectric",
    "matbench_elastic",
    "matbench_expt_gap",
    "matbench_glass",
    "matbench_jdft2d",
    "matbench_perovskites",
    "matbench_phonons",
    "matbench_steels",
]
CLASSIFICATION_TASKS = {
    "matbench_expt_is_metal",
    "matbench_glass",
    "matbench_mp_is_metal",
}
ELASTIC_MULTITASK_TASK = "matbench_elastic"
ELASTIC_TARGET_TASKS = ("matbench_log_gvrh", "matbench_log_kvrh")
OFFICIAL_FIT_SETTINGS = {
    "increase_bs": False,
    "lr": 0.005,
    "epochs": 50,
    "act": "elu",
    "out_act": "relu",
    "batch_size": 32,
    "loss": "mae",
    "xscale": "minmax",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official MODNet v0.1.12 Matbench pipeline and optionally "
            "export its fold-level selected descriptors for KAN experiments."
        )
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--task-set", choices=["small", "all"], default="small")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--fast", action="store_true", help="Official MODNet debug mode.")
    parser.add_argument("--nested-folds", type=int, default=5)
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--hp-strategy", choices=["fit_preset", "ga"], default="fit_preset")
    parser.add_argument("--no-hp-optimization", action="store_true")
    parser.add_argument("--no-inner-feat-selection", action="store_true")
    parser.add_argument("--no-use-precomputed-cross-nmi", action="store_true")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--force-featurize", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose requested official outputs already exist.",
    )
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--save-final-model", action="store_true")
    parser.add_argument("--save-folds", action="store_true")
    parser.add_argument("--export-feature-folds", action="store_true")
    parser.add_argument(
        "--export-max-features",
        type=int,
        default=512,
        help="Maximum ranked official MODNet descriptors to export per fold; 0 exports all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_official_modnet() -> None:
    try:
        import modnet  # noqa: F401
        import tensorflow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Official MODNet reproduction requires a separate Python 3.8 environment "
            "with modnet==0.1.12 and TensorFlow installed. Create it from "
            "environment-modnet-v012.yml, then rerun this script there. "
            f"Missing import: {exc}"
        ) from exc


def setup_threading() -> None:
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def normalize_task(task: str) -> str:
    task = task.strip()
    if not task:
        raise ValueError("empty task name")
    return task if task.startswith("matbench_") else f"matbench_{task}"


def selected_tasks(args: argparse.Namespace) -> list[str]:
    if args.tasks:
        return [normalize_task(task) for task in args.tasks]
    return list(MATBENCH_TASKS if args.task_set == "all" else SMALL_MATBENCH_TASKS)


def task_outputs_complete(output_dir: Path, task: str, require_feature_folds: bool) -> bool:
    task_dir = output_dir / task
    if not (task_dir / "official-modnet-run-metadata.json").exists():
        return False
    if not require_feature_folds:
        return True
    feature_dir = task_dir / "official_feature_folds"
    for fold in range(5):
        fold_dir = feature_dir / f"fold_{fold}"
        if not (
            (fold_dir / "train_features.csv.gz").exists()
            and (fold_dir / "test_features.csv.gz").exists()
            and (fold_dir / "feature_order.json").exists()
        ):
            return False
    return True


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        column: str(column).replace(" ", "_").replace("(", "").replace(")", "")
        for column in df.columns
    }
    return df.rename(columns=mapping)


def task_is_classification(task: str, targets: pd.DataFrame) -> bool:
    if task in CLASSIFICATION_TASKS:
        return True
    if len(targets.columns) != 1:
        return False
    values = set(pd.Series(targets.iloc[:, 0]).dropna().astype(int).unique())
    return values.issubset({0, 1}) and ("is_metal" in task or "glass" in task)


def load_or_featurize(
    task: str,
    cache_dir: Path,
    n_jobs: int,
    force_featurize: bool,
) -> tuple[Any, dict[str, Any]]:
    from matminer.datasets import load_dataset
    from modnet.featurizers.presets import CompositionOnlyFeaturizer, DeBreuck2020Featurizer
    from modnet.preprocessing import MODData
    from pymatgen.core import Composition

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{task}_moddata.pkl.gz"
    if cache_path.exists() and not force_featurize:
        data = MODData.load(cache_path)
        targets = data.df_targets.copy()
        return data, {
            "task": task,
            "target_names": list(targets.columns),
            "classification": task in CLASSIFICATION_TASKS,
            "source_tasks": list(ELASTIC_TARGET_TASKS) if task == ELASTIC_MULTITASK_TASK else [task],
            "cache_path": str(cache_path),
            "loaded_from_cache": True,
        }

    df = load_elastic_multitask_dataframe() if task == ELASTIC_MULTITASK_TASK else sanitize_columns(load_dataset(task))
    target_names = [
        column for column in df.columns if column not in ("id", "structure", "composition")
    ]
    if not target_names:
        raise RuntimeError(f"No target columns found for {task}")
    targets = df[target_names]
    classification = task_is_classification(task, targets)
    num_classes = {name: 2 for name in target_names} if classification else None

    if "structure" in df.columns:
        materials = df["structure"].tolist()
        featurizer = DeBreuck2020Featurizer(fast_oxid=True)
        input_type = "structure"
    elif "composition" in df.columns:
        materials = df["composition"].map(Composition).tolist()
        featurizer = CompositionOnlyFeaturizer()
        input_type = "composition"
    else:
        raise RuntimeError(f"{task} has neither structure nor composition column")

    data = MODData(
        materials=materials,
        targets=targets.values,
        target_names=target_names,
        num_classes=num_classes,
        featurizer=featurizer,
    )
    data.featurize(n_jobs=n_jobs)
    data.save(cache_path)
    return data, {
        "task": task,
        "target_names": target_names,
        "classification": classification,
        "input_type": input_type,
        "source_tasks": list(ELASTIC_TARGET_TASKS) if task == ELASTIC_MULTITASK_TASK else [task],
        "cache_path": str(cache_path),
        "loaded_from_cache": False,
    }


def load_elastic_multitask_dataframe() -> Any:
    import pandas as pd
    from matminer.datasets import load_dataset

    frames = [sanitize_columns(load_dataset(task)) for task in ELASTIC_TARGET_TASKS]
    base = frames[0].copy()
    for frame in frames[1:]:
        if len(frame) != len(base):
            raise RuntimeError("Elastic Matbench task sizes differ; cannot form multitask target")
        if "structure" in base.columns and "structure" in frame.columns:
            formulas_a = [structure.composition.reduced_formula for structure in base["structure"]]
            formulas_b = [structure.composition.reduced_formula for structure in frame["structure"]]
            if formulas_a != formulas_b:
                raise RuntimeError("Elastic Matbench structures are not aligned")
        target_columns = [
            column for column in frame.columns if column not in ("id", "structure", "composition")
        ]
        if len(target_columns) != 1:
            raise RuntimeError(f"Expected one target column in elastic source task, got {target_columns}")
        base[target_columns[0]] = frame[target_columns[0]].to_numpy()
    return pd.DataFrame(base)


def run_official_benchmark(
    data: Any,
    classification: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from modnet.matbench.benchmark import matbench_benchmark
    from modnet.models import EnsembleMODNetModel

    target_names = list(data.df_targets.columns)
    names = [[target_names]]
    weights = {name: 1 for name in target_names}
    kwargs = {
        "model_type": EnsembleMODNetModel,
        "n_models": args.n_models,
        "classification": classification,
        "fast": args.fast,
        "nested": 0 if args.fast else args.nested_folds,
        "n_jobs": args.n_jobs,
        "save_folds": args.save_folds,
        "save_models": args.save_final_model,
        "hp_optimization": not args.no_hp_optimization,
        "hp_strategy": args.hp_strategy,
        "inner_feat_selection": not args.no_inner_feat_selection,
        "use_precomputed_cross_nmi": not args.no_use_precomputed_cross_nmi,
    }
    if "random_state" in inspect.signature(matbench_benchmark).parameters:
        kwargs["random_state"] = args.random_state
    return matbench_benchmark(
        data,
        names,
        weights,
        dict(OFFICIAL_FIT_SETTINGS),
        **kwargs,
    )


def class_one_probability(predictions: pd.DataFrame, target_name: str) -> np.ndarray:
    candidates = [
        f"{target_name}_prob_1",
        f"{target_name}_1",
        target_name,
    ]
    for column in candidates:
        if column in predictions.columns:
            return predictions[column].to_numpy(dtype=float)
    if predictions.shape[1] == 2:
        return predictions.iloc[:, 1].to_numpy(dtype=float)
    return predictions.iloc[:, 0].to_numpy(dtype=float)


def fold_metrics(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    classification: bool,
) -> dict[str, float]:
    if classification:
        target_name = str(targets.columns[0])
        y_true = targets[target_name].to_numpy()
        y_prob = class_one_probability(predictions, target_name)
        y_pred = (y_prob >= 0.5).astype(int)
        return {
            "rocauc": float(roc_auc_score(y_true.astype(int), y_prob)),
            "accuracy": float(accuracy_score(y_true.astype(int), y_pred)),
            "ap_score": float(average_precision_score(y_true.astype(int), y_prob)),
        }

    metrics = {}
    maes = []
    rmses = []
    max_errors = []
    for target_idx, target_name in enumerate(targets.columns):
        safe_target = str(target_name).replace(" ", "_").replace("(", "").replace(")", "")
        y_true = targets[target_name].to_numpy()
        if target_name in predictions.columns:
            y_pred = predictions[target_name].to_numpy(dtype=float)
        elif target_idx < predictions.shape[1]:
            y_pred = predictions.iloc[:, target_idx].to_numpy(dtype=float)
        else:
            y_pred = predictions.iloc[:, 0].to_numpy(dtype=float)
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        err = float(max_error(y_true, y_pred))
        maes.append(mae)
        rmses.append(rmse)
        max_errors.append(err)
        metrics[f"mae__{safe_target}"] = mae
        metrics[f"rmse__{safe_target}"] = rmse
        metrics[f"max_error__{safe_target}"] = err
    metrics["mae"] = float(np.mean(maes))
    metrics["rmse"] = float(np.mean(rmses))
    metrics["max_error"] = float(np.mean(max_errors))
    return metrics


def summarize_results(
    task: str,
    results: dict[str, Any],
    classification: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_rows = []
    for fold, (predictions, targets) in enumerate(
        zip(results.get("predictions", []), results.get("targets", []))
    ):
        metrics = fold_metrics(predictions, targets, classification)
        row: dict[str, Any] = {
            "task": task,
            "fold": fold,
            "task_type": "classification" if classification else "regression",
            "n_test": len(targets),
        }
        row.update(metrics)
        if "scores" in results and fold < len(results["scores"]):
            score = results["scores"][fold]
            if isinstance(score, np.ndarray):
                score = float(np.mean(score))
            row["official_score"] = float(score)
        if "best_presets" in results and fold < len(results["best_presets"]):
            row["best_preset"] = json.dumps(results["best_presets"][fold], sort_keys=True)
        fold_rows.append(row)

    metric_names = ["rocauc", "accuracy", "ap_score"] if classification else sorted(
        key for row in fold_rows for key in row if key.startswith(("mae", "rmse", "max_error"))
    )
    summary: dict[str, Any] = {
        "task": task,
        "task_type": "classification" if classification else "regression",
        "folds": len(fold_rows),
    }
    for metric in metric_names:
        values = [row[metric] for row in fold_rows if metric in row and np.isfinite(row[metric])]
        summary[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
        summary[f"{metric}_std"] = float(np.std(values)) if values else float("nan")
    return fold_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results_pickle(path: Path, results: dict[str, Any], include_models: bool) -> None:
    safe_keys = [
        "targets",
        "predictions",
        "errors",
        "scores",
        "nested_learning_curves",
        "best_learning_curves",
        "best_presets",
        "nested_losses",
        "stds",
    ]
    payload = {key: results[key] for key in safe_keys if key in results}
    if include_models and "model" in results:
        payload["model"] = results["model"]
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle)


def export_feature_folds(
    data: Any,
    task_dir: Path,
    task: str,
    classification: bool,
    max_features: int,
    n_jobs: int,
    inner_feat_selection: bool,
    use_precomputed_cross_nmi: bool,
) -> Path:
    from modnet.matbench.benchmark import matbench_kfold_splits
    from modnet.preprocessing import MODData

    export_dir = task_dir / "official_feature_folds"
    export_dir.mkdir(parents=True, exist_ok=True)
    for fold, (train_idx, test_idx) in enumerate(
        matbench_kfold_splits(data, classification=classification)
    ):
        train_data, test_data = data.split((train_idx, test_idx))
        saved_train = task_dir / "folds" / f"train_moddata_f{fold + 1}"
        if saved_train.exists():
            train_data = MODData.load(saved_train)
        elif inner_feat_selection:
            train_data.feature_selection(
                n=-1,
                use_precomputed_cross_nmi=use_precomputed_cross_nmi,
                n_jobs=n_jobs,
            )
            saved_train.parent.mkdir(parents=True, exist_ok=True)
            train_data.save(saved_train)

        feature_order = list(
            getattr(train_data, "optimal_features", None) or train_data.df_featurized.columns
        )
        limit = len(feature_order) if max_features <= 0 else min(max_features, len(feature_order))
        selected = feature_order[:limit]
        fold_dir = export_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_data.df_featurized.reindex(columns=selected).to_csv(
            fold_dir / "train_features.csv.gz",
            compression="gzip",
        )
        test_data.df_featurized.reindex(columns=selected).to_csv(
            fold_dir / "test_features.csv.gz",
            compression="gzip",
        )
        train_data.df_targets.to_csv(fold_dir / "train_targets.csv.gz", compression="gzip")
        test_data.df_targets.to_csv(fold_dir / "test_targets.csv.gz", compression="gzip")
        (fold_dir / "feature_order.json").write_text(
            json.dumps(selected, indent=2),
            encoding="utf-8",
        )
        (fold_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "fold": fold,
                    "classification": classification,
                    "n_features_exported": len(selected),
                    "source": "official MODNet v0.1.12 feature_selection",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return export_dir


def run_task(args: argparse.Namespace, task: str, output_dir: Path) -> dict[str, Any]:
    task_dir = (output_dir / task).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else task_dir / "precomputed"
    cache_dir = cache_dir.resolve()
    start = time.perf_counter()
    data, metadata = load_or_featurize(
        task,
        cache_dir=cache_dir,
        n_jobs=args.n_jobs,
        force_featurize=args.force_featurize,
    )
    metadata["modnet_version_target"] = "0.1.12"
    metadata["official_fit_settings"] = OFFICIAL_FIT_SETTINGS
    metadata["official_matbench_settings"] = {
        "target_hierarchy": [[list(data.df_targets.columns)]],
        "target_weights": {name: 1 for name in data.df_targets.columns},
        "model_type": "EnsembleMODNetModel",
        "n_models": args.n_models,
        "hp_optimization": not args.no_hp_optimization,
        "hp_strategy": args.hp_strategy,
        "inner_feat_selection": not args.no_inner_feat_selection,
        "use_precomputed_cross_nmi": not args.no_use_precomputed_cross_nmi,
        "nested": 0 if args.fast else args.nested_folds,
        "save_folds": args.save_folds,
        "save_models": args.save_final_model,
        "random_state": args.random_state,
    }

    old_cwd = Path.cwd()
    results: dict[str, Any] | None = None
    try:
        os.chdir(task_dir)
        if not args.skip_benchmark:
            results = run_official_benchmark(
                data,
                classification=bool(metadata["classification"]),
                args=args,
            )
            fold_rows, summary = summarize_results(
                task,
                results,
                classification=bool(metadata["classification"]),
            )
            summary.update(
                {
                    "fast": args.fast,
                    "nested_folds": 0 if args.fast else args.nested_folds,
                    "n_models": args.n_models,
                    "seconds": time.perf_counter() - start,
                }
            )
            write_csv(task_dir / f"official-modnet-fold-results-{task}.csv", fold_rows)
            write_csv(task_dir / f"official-modnet-summary-{task}.csv", [summary])
            (task_dir / f"official-modnet-summary-{task}.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            save_results_pickle(
                task_dir / f"official-modnet-results-{task}.pkl.gz",
                results,
                include_models=args.save_final_model,
            )
        else:
            summary = {
                "task": task,
                "task_type": "classification" if metadata["classification"] else "regression",
                "skipped_benchmark": True,
                "seconds": time.perf_counter() - start,
            }

        if args.export_feature_folds:
            feature_dir = export_feature_folds(
                data,
                task_dir=task_dir,
                task=task,
                classification=bool(metadata["classification"]),
                max_features=args.export_max_features,
                n_jobs=args.n_jobs,
                inner_feat_selection=not args.no_inner_feat_selection,
                use_precomputed_cross_nmi=not args.no_use_precomputed_cross_nmi,
            )
            summary["official_feature_folds"] = str(feature_dir)

    finally:
        os.chdir(old_cwd)

    metadata.update(summary)
    (task_dir / "official-modnet-run-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    tasks = selected_tasks(args)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("benchmarks") / f"official-modnet-v012-{time.strftime('%Y%m%d-%H%M%S')}"
    ).resolve()
    print(f"Tasks: {tasks}", flush=True)
    print(f"Output dir: {output_dir.resolve()}", flush=True)
    if args.dry_run:
        return

    require_official_modnet()
    setup_threading()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for task in tasks:
        print(f"\n=== Official MODNet v0.1.12: {task} ===", flush=True)
        try:
            if args.skip_existing and task_outputs_complete(
                output_dir, task, require_feature_folds=args.export_feature_folds
            ):
                metadata_path = output_dir / task / "official-modnet-run-metadata.json"
                summary = json.loads(metadata_path.read_text(encoding="utf-8"))
                summary["skipped_existing"] = True
                summaries.append(summary)
                print(json.dumps(summary, indent=2), flush=True)
                continue
            summary = run_task(args, task, output_dir)
            summaries.append(summary)
            print(json.dumps(summary, indent=2), flush=True)
        except Exception:
            print_exc()
            summaries.append({"task": task, "failed": True})
    write_csv(output_dir / "official-modnet-all-summary.csv", summaries)
    with gzip.open(output_dir / "official-modnet-all-summary.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"\nWrote {output_dir / 'official-modnet-all-summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
