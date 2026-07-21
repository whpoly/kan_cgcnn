from __future__ import annotations

import argparse
import csv
import gzip
import inspect
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Set native-library limits before NumPy, TensorFlow, matminer, or MODNet are
# imported. Setting these after importing TensorFlow is too late and can cause
# severe oversubscription in nested Matbench/feature-selection jobs.
for _thread_env_name in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
):
    os.environ[_thread_env_name] = os.environ.get("MODNET_THREADS_PER_PROCESS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

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
    parser.add_argument(
        "--task-timeout-minutes",
        type=float,
        default=1440.0,
        help="Hard timeout for each isolated task; 0 disables the timeout.",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--max-task-attempts", type=int, default=2)
    parser.add_argument(
        "--retry-n-jobs",
        type=int,
        default=1,
        help="Use this safer worker count after an isolated task fails or times out.",
    )
    parser.add_argument("--worker-task", default=None, help=argparse.SUPPRESS)
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
    if args.worker_task:
        return [normalize_task(args.worker_task)]
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
        required = (
            fold_dir / "train_features.csv.gz",
            fold_dir / "test_features.csv.gz",
            fold_dir / "train_targets.csv.gz",
            fold_dir / "test_targets.csv.gz",
            fold_dir / "feature_order.json",
            fold_dir / "metadata.json",
        )
        if not all(path.exists() and path.stat().st_size > 0 for path in required):
            return False
    return True


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def write_task_status(task_dir: Path, task: str, stage: str, **details: Any) -> None:
    atomic_write_json(
        task_dir / "STATUS.json",
        {
            "task": task,
            "stage": stage,
            "pid": os.getpid(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **details,
        },
    )


def atomic_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp_path, compression="gzip")
    os.replace(temp_path, path)


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
        try:
            data = MODData.load(cache_path)
        except Exception as exc:
            print(
                f"Ignoring unreadable featurization cache {cache_path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        else:
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
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


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
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temp_path, "wb") as handle:
        pickle.dump(payload, handle)
    os.replace(temp_path, path)


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
        fold_dir = export_dir / f"fold_{fold}"
        complete_paths = (
            fold_dir / "train_features.csv.gz",
            fold_dir / "test_features.csv.gz",
            fold_dir / "train_targets.csv.gz",
            fold_dir / "test_targets.csv.gz",
            fold_dir / "feature_order.json",
            fold_dir / "metadata.json",
        )
        if all(path.exists() and path.stat().st_size > 0 for path in complete_paths):
            print(f"Feature fold {fold} already complete; reusing it.", flush=True)
            continue
        write_task_status(task_dir, task, "feature_selection", fold=fold)
        train_data, test_data = data.split((train_idx, test_idx))
        saved_train = task_dir / "folds" / f"train_moddata_f{fold + 1}"
        loaded_saved_train = False
        if saved_train.exists():
            try:
                train_data = MODData.load(saved_train)
            except Exception as exc:
                print(
                    f"Ignoring unreadable saved fold {saved_train}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            else:
                loaded_saved_train = True
        if not loaded_saved_train and inner_feat_selection:
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
        fold_dir.mkdir(parents=True, exist_ok=True)
        atomic_dataframe_csv(
            train_data.df_featurized.reindex(columns=selected),
            fold_dir / "train_features.csv.gz",
        )
        atomic_dataframe_csv(
            test_data.df_featurized.reindex(columns=selected),
            fold_dir / "test_features.csv.gz",
        )
        atomic_dataframe_csv(train_data.df_targets, fold_dir / "train_targets.csv.gz")
        atomic_dataframe_csv(test_data.df_targets, fold_dir / "test_targets.csv.gz")
        atomic_write_json(fold_dir / "feature_order.json", selected)
        # Metadata is written last and therefore acts as the fold completion marker.
        atomic_write_json(
            fold_dir / "metadata.json",
            {
                "task": task,
                "fold": fold,
                "classification": classification,
                "n_features_exported": len(selected),
                "source": "official MODNet v0.1.12 feature_selection",
            },
        )
    return export_dir


def run_task(args: argparse.Namespace, task: str, output_dir: Path) -> dict[str, Any]:
    task_dir = (output_dir / task).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else task_dir / "precomputed"
    cache_dir = cache_dir.resolve()
    start = time.perf_counter()
    write_task_status(task_dir, task, "featurization", n_jobs=args.n_jobs)
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
        existing_summary_path = task_dir / f"official-modnet-summary-{task}.json"
        existing_benchmark_paths = (
            existing_summary_path,
            task_dir / f"official-modnet-fold-results-{task}.csv",
            task_dir / f"official-modnet-results-{task}.pkl.gz",
        )
        reuse_existing_benchmark = args.skip_existing and all(
            path.exists() and path.stat().st_size > 0 for path in existing_benchmark_paths
        )
        if reuse_existing_benchmark and not args.skip_benchmark:
            summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
            summary["reused_existing_benchmark"] = True
            print("Official benchmark outputs already complete; exporting missing folds only.", flush=True)
        elif not args.skip_benchmark:
            write_task_status(task_dir, task, "official_benchmark", n_jobs=args.n_jobs)
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
            atomic_write_json(task_dir / f"official-modnet-summary-{task}.json", summary)
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
            write_task_status(task_dir, task, "feature_export", n_jobs=args.n_jobs)
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
    atomic_write_json(task_dir / "official-modnet-run-metadata.json", metadata)
    write_task_status(
        task_dir,
        task,
        "complete",
        seconds=time.perf_counter() - start,
    )
    return metadata


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def worker_command(
    args: argparse.Namespace,
    task: str,
    output_dir: Path,
    n_jobs: int,
) -> list[str]:
    # Appending repeated scalar options is intentional: argparse uses the last
    # value, so the resolved output directory and safe retry worker count win.
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker-task",
        task,
        "--output-dir",
        str(output_dir),
        "--n-jobs",
        str(n_jobs),
    ]


def status_stage(task_dir: Path) -> str:
    status_path = task_dir / "STATUS.json"
    if not status_path.exists():
        return "starting"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        fold = payload.get("fold")
        return f"{payload.get('stage', 'unknown')}" + (f" fold={fold}" if fold is not None else "")
    except (OSError, ValueError):
        return "status-unreadable"


def run_isolated_task(
    args: argparse.Namespace,
    task: str,
    output_dir: Path,
) -> dict[str, Any]:
    task_dir = output_dir / task
    attempts = max(1, int(args.max_task_attempts))
    failures: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        if task_outputs_complete(output_dir, task, require_feature_folds=args.export_feature_folds):
            return json.loads(
                (task_dir / "official-modnet-run-metadata.json").read_text(encoding="utf-8")
            )
        n_jobs = args.n_jobs if attempt == 1 else max(1, int(args.retry_n_jobs))
        cmd = worker_command(args, task, output_dir, n_jobs=n_jobs)
        print(
            f"Starting isolated task {task}, attempt {attempt}/{attempts}, n_jobs={n_jobs}",
            flush=True,
        )
        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(cmd, **popen_kwargs)
        started = time.monotonic()
        heartbeat = max(1.0, float(args.heartbeat_seconds))
        timeout_seconds = (
            float(args.task_timeout_minutes) * 60
            if args.task_timeout_minutes > 0
            else None
        )
        timed_out = False
        try:
            while True:
                try:
                    return_code = process.wait(timeout=heartbeat)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(
                        f"[heartbeat] task={task} attempt={attempt} "
                        f"elapsed_min={elapsed / 60:.1f} stage={status_stage(task_dir)}",
                        flush=True,
                    )
                    if timeout_seconds is not None and elapsed >= timeout_seconds:
                        timed_out = True
                        terminate_process_tree(process)
                        return_code = process.returncode if process.returncode is not None else -9
                        break
        except KeyboardInterrupt:
            terminate_process_tree(process)
            raise

        failure = {
            "task": task,
            "attempt": attempt,
            "n_jobs": n_jobs,
            "timed_out": timed_out,
            "return_code": return_code,
            "stage": status_stage(task_dir),
            "elapsed_seconds": time.monotonic() - started,
        }
        failures.append(failure)
        write_task_status(
            task_dir,
            task,
            "retry_pending",
            **{key: value for key, value in failure.items() if key not in {"task", "stage"}},
        )
        print(f"Isolated task failed: {json.dumps(failure)}", flush=True)

    failure_summary = {
        "task": task,
        "failed": True,
        "attempts": failures,
    }
    atomic_write_json(task_dir / "FAILED.json", failure_summary)
    write_task_status(task_dir, task, "failed", attempts=failures)
    return failure_summary


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

    setup_threading()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_task:
        require_official_modnet()
        task = normalize_task(args.worker_task)
        try:
            summary = run_task(args, task, output_dir)
        except Exception as exc:
            write_task_status(
                output_dir / task,
                task,
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        print(json.dumps(summary, indent=2), flush=True)
        return

    summaries = []
    failed = False
    for task in tasks:
        print(f"\n=== Official MODNet v0.1.12: {task} ===", flush=True)
        if args.skip_existing and task_outputs_complete(
            output_dir, task, require_feature_folds=args.export_feature_folds
        ):
            metadata_path = output_dir / task / "official-modnet-run-metadata.json"
            summary = json.loads(metadata_path.read_text(encoding="utf-8"))
            summary["skipped_existing"] = True
            summaries.append(summary)
            print(json.dumps(summary, indent=2), flush=True)
            continue
        summary = run_isolated_task(args, task, output_dir)
        summaries.append(summary)
        failed = failed or bool(summary.get("failed"))
        print(json.dumps(summary, indent=2), flush=True)
    write_csv(output_dir / "official-modnet-all-summary.csv", summaries)
    with gzip.open(output_dir / "official-modnet-all-summary.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"\nWrote {output_dir / 'official-modnet-all-summary.csv'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
