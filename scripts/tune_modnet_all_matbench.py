from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_FAMILIES = [
    "mlp",
    "hybrid-fastkan",
    "hybrid-spline",
    "direct-fastkan",
    "direct-spline",
    "fastkan",
    "spline",
]
DEFAULT_MODEL_FAMILIES = ["mlp", "fastkan", "spline"]
TASK_TYPES = ["regression", "classification"]
FEATURE_PRESETS = [
    "auto",
    "pymatgen-composition",
    "pymatgen-structure",
    "matminer-composition",
    "matminer-structure-lite",
]
MAXIMIZE_METRICS = {
    "best_val_r2",
    "best_val_accuracy",
    "best_val_balanced_accuracy",
    "best_val_f1",
    "best_val_rocauc",
    "test_r2",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_f1",
    "test_rocauc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Matbench-strict MODNet-style tuning/final benchmarks on every "
            "supported small regression/classification dataset, with per-dataset leaderboard summaries."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to run. Defaults to structure/composition Matbench v0.1 tasks under --max-samples.",
    )
    parser.add_argument("--skip-datasets", nargs="*", default=[])
    parser.add_argument("--task-types", nargs="+", choices=TASK_TYPES, default=TASK_TYPES)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument(
        "--model-families",
        nargs="+",
        choices=MODEL_FAMILIES,
        default=DEFAULT_MODEL_FAMILIES,
    )
    parser.add_argument(
        "--protocol",
        choices=["matbench-nested", "legacy-global"],
        default="matbench-nested",
    )
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--tune-folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--final-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--search-space", choices=["compact", "random", "grid"], default="compact")
    parser.add_argument("--num-random-trials", type=int, default=8)
    parser.add_argument("--max-trials-per-family", type=int, default=None)
    parser.add_argument(
        "--metric",
        choices=[
            "auto",
            "best_val_mae",
            "best_val_rmse",
            "best_val_r2",
            "best_val_accuracy",
            "best_val_balanced_accuracy",
            "best_val_f1",
            "best_val_rocauc",
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1",
            "test_rocauc",
        ],
        default="auto",
    )
    parser.add_argument("--featurizer-preset", choices=FEATURE_PRESETS, default="auto")
    parser.add_argument("--featurizer-jobs", type=int, default=1)
    parser.add_argument("--n-feature-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--common-dim-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--group-dim-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--property-dim-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--target-dim-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--kan-grid-size-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--kan-spline-order-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--lr-candidates", type=float, nargs="+", default=None)
    parser.add_argument("--weight-decay-candidates", type=float, nargs="+", default=None)
    parser.add_argument("--dropout-candidates", type=float, nargs="+", default=None)
    parser.add_argument("--loss-candidates", nargs="+", choices=["mae", "rmse", "mse", "bce"], default=["mae"])
    parser.add_argument("--prune-kan-fraction-candidates", type=float, nargs="+", default=[0.0])
    parser.add_argument("--prune-mode", choices=["edge", "parameter"], default="edge")
    parser.add_argument("--prune-finetune-epochs", type=int, default=0)
    parser.add_argument("--posthoc-prune-kan-fraction", type=float, default=0.3)
    parser.add_argument("--kan-l1-lambda", type=float, default=0.0)
    parser.add_argument("--kan-sparsity-mode", choices=["edge-group", "parameter-l1"], default="edge-group")
    parser.add_argument("--posthoc-kan-sparsity-lambda", type=float, default=1e-4)
    parser.add_argument("--activation", choices=["relu", "elu", "silu"], default="elu")
    parser.add_argument("--trial-timeout-minutes", type=float, default=720.0)
    parser.add_argument("--allow-kan-larger-than-mlp", action="store_true")
    parser.add_argument("--scaler", choices=["minmax", "standard", "none"], default="minmax")
    parser.add_argument("--target-scale", choices=["none", "standard"], default="none")
    parser.add_argument("--impute-strategy", default="median")
    parser.add_argument("--tune-epochs", type=int, default=1000)
    parser.add_argument("--final-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=100)
    parser.add_argument("--tune-train-size", type=int, default=None)
    parser.add_argument(
        "--tune-test-size",
        type=int,
        default=256,
        help="Only used when --evaluate-tune-test is set.",
    )
    parser.add_argument("--evaluate-tune-test", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--log-every-epochs", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-export-final-formulas", action="store_true")
    parser.add_argument("--formula-top-k", type=int, default=40)
    parser.add_argument("--formula-min-abs", type=float, default=0.0)
    parser.add_argument("--simple-formula-min-inputs", type=int, default=5)
    parser.add_argument("--simple-formula-max-inputs", type=int, default=10)
    parser.add_argument("--simple-formula-max-terms", type=int, default=10)
    parser.add_argument("--simple-formula-coverage", type=float, default=0.95)
    parser.add_argument("--simple-formula-calibration-ratio", type=float, default=0.1)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip datasets that already have a summary CSV.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def matbench_metadata() -> dict[str, Any]:
    from matbench.metadata import mbv01_metadata

    return dict(mbv01_metadata)


def is_supported(meta: Any, args: argparse.Namespace | None = None) -> bool:
    task_types = set(args.task_types) if args is not None else set(TASK_TYPES)
    max_samples = args.max_samples if args is not None else 20000
    return (
        meta.task_type in task_types
        and meta.input_type in ("structure", "composition")
        and int(meta.n_samples) <= int(max_samples)
    )


def supported_datasets(metadata: dict[str, Any], args: argparse.Namespace | None = None) -> list[str]:
    datasets = [name for name, meta in metadata.items() if is_supported(meta, args)]
    if "matbench_log_gvrh" in datasets and "matbench_log_kvrh" in datasets:
        datasets = [
            name
            for name in datasets
            if name not in {"matbench_log_gvrh", "matbench_log_kvrh"}
        ]
        datasets.append("matbench_elastic")
    return sorted(datasets)


def metadata_row(dataset: str, meta: Any) -> dict[str, Any]:
    if dataset == "matbench_elastic":
        return {
            "dataset": dataset,
            "task_type": "regression",
            "input_type": "structure",
            "target": "log10(G_VRH)+log10(K_VRH)",
            "unit": None,
            "n_samples": int(meta.n_samples),
            "mad": None,
            "frac_true": None,
        }
    return {
        "dataset": dataset,
        "task_type": meta.task_type,
        "input_type": meta.input_type,
        "target": meta.target,
        "unit": getattr(meta, "unit", None),
        "n_samples": int(meta.n_samples),
        "mad": getattr(meta, "mad", None),
        "frac_true": getattr(meta, "frac_true", None),
    }


def metadata_for_dataset(dataset: str, metadata: dict[str, Any]) -> Any:
    if dataset == "matbench_elastic":
        return metadata["matbench_log_gvrh"]
    return metadata[dataset]


def select_datasets(args: argparse.Namespace, metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    all_supported = supported_datasets(metadata, args)
    requested = args.datasets or all_supported
    unknown = [dataset for dataset in requested if dataset not in metadata and dataset != "matbench_elastic"]
    if unknown:
        raise ValueError(f"Unknown Matbench dataset(s): {', '.join(unknown)}")

    unsupported = [
        dataset
        for dataset in requested
        if dataset != "matbench_elastic" and not is_supported(metadata[dataset], args)
    ]
    if unsupported:
        raise ValueError(
            "Unsupported by this small Matbench MODNet runner: "
            + ", ".join(
                f"{dataset}({metadata[dataset].task_type}/{metadata[dataset].input_type}, "
                f"n={metadata[dataset].n_samples})"
                for dataset in unsupported
            )
        )

    skipped = [dataset for dataset in requested if dataset in set(args.skip_datasets)]
    selected = [dataset for dataset in requested if dataset not in set(args.skip_datasets)]
    return selected, skipped


def build_tune_command(args: argparse.Namespace, dataset: str, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "tune_modnet_kan.py"),
        "--dataset",
        dataset,
        "--output-dir",
        str(output_dir),
        "--search-space",
        args.search_space,
        "--num-random-trials",
        str(args.num_random_trials),
        "--protocol",
        args.protocol,
        "--inner-folds",
        str(args.inner_folds),
        "--metric",
        args.metric,
        "--featurizer-preset",
        args.featurizer_preset,
        "--featurizer-jobs",
        str(args.featurizer_jobs),
        "--scaler",
        args.scaler,
        "--target-scale",
        args.target_scale,
        "--impute-strategy",
        args.impute_strategy,
        "--tune-epochs",
        str(args.tune_epochs),
        "--final-epochs",
        str(args.final_epochs),
        "--batch-size",
        str(args.batch_size),
        "--val-ratio",
        str(args.val_ratio),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--log-every-epochs",
        str(args.log_every_epochs),
        "--prune-mode",
        args.prune_mode,
        "--prune-finetune-epochs",
        str(args.prune_finetune_epochs),
        "--posthoc-prune-kan-fraction",
        str(args.posthoc_prune_kan_fraction),
        "--kan-l1-lambda",
        str(args.kan_l1_lambda),
        "--kan-sparsity-mode",
        args.kan_sparsity_mode,
        "--posthoc-kan-sparsity-lambda",
        str(args.posthoc_kan_sparsity_lambda),
        "--activation",
        args.activation,
        "--trial-timeout-minutes",
        str(args.trial_timeout_minutes),
        "--simple-formula-min-inputs",
        str(args.simple_formula_min_inputs),
        "--simple-formula-max-inputs",
        str(args.simple_formula_max_inputs),
        "--simple-formula-max-terms",
        str(args.simple_formula_max_terms),
        "--simple-formula-coverage",
        str(args.simple_formula_coverage),
        "--simple-formula-calibration-ratio",
        str(args.simple_formula_calibration_ratio),
    ]
    if args.tune_train_size is not None:
        cmd.extend(["--tune-train-size", str(args.tune_train_size)])
    extend_values(cmd, "--model-families", args.model_families)
    extend_values(cmd, "--tune-folds", args.tune_folds)
    extend_values(cmd, "--final-folds", args.final_folds)
    extend_optional_values(cmd, "--n-feature-candidates", args.n_feature_candidates)
    extend_optional_values(cmd, "--common-dim-candidates", args.common_dim_candidates)
    extend_optional_values(cmd, "--group-dim-candidates", args.group_dim_candidates)
    extend_optional_values(cmd, "--property-dim-candidates", args.property_dim_candidates)
    extend_optional_values(cmd, "--target-dim-candidates", args.target_dim_candidates)
    extend_optional_values(cmd, "--kan-grid-size-candidates", args.kan_grid_size_candidates)
    extend_optional_values(cmd, "--kan-spline-order-candidates", args.kan_spline_order_candidates)
    extend_optional_values(cmd, "--lr-candidates", args.lr_candidates)
    extend_optional_values(cmd, "--weight-decay-candidates", args.weight_decay_candidates)
    extend_optional_values(cmd, "--dropout-candidates", args.dropout_candidates)
    extend_values(cmd, "--loss-candidates", args.loss_candidates)
    extend_values(cmd, "--prune-kan-fraction-candidates", args.prune_kan_fraction_candidates)
    if args.max_trials_per_family is not None:
        cmd.extend(["--max-trials-per-family", str(args.max_trials_per_family)])
    if args.allow_kan_larger_than_mlp:
        cmd.append("--allow-kan-larger-than-mlp")
    if args.evaluate_tune_test:
        cmd.extend(["--evaluate-tune-test", "--tune-test-size", str(args.tune_test_size)])
    if args.no_export_final_formulas:
        cmd.append("--no-export-final-formulas")
    cmd.extend(["--formula-top-k", str(args.formula_top_k)])
    cmd.extend(["--formula-min-abs", str(args.formula_min_abs)])
    if args.require_cuda:
        cmd.append("--require-cuda")
    if args.skip_final:
        cmd.append("--skip-final")
    if args.resume:
        cmd.append("--resume")
    if args.fail_fast:
        cmd.append("--fail-fast")
    return cmd


def extend_values(cmd: list[str], flag: str, values: list[Any]) -> None:
    cmd.append(flag)
    cmd.extend(str(value) for value in values)


def extend_optional_values(cmd: list[str], flag: str, values: list[Any] | None) -> None:
    if values is not None:
        extend_values(cmd, flag, values)


def summary_csv_path(dataset_dir: Path, dataset: str, skip_final: bool) -> Path:
    final_path = dataset_dir / f"final-summary-{dataset}.csv"
    tuning_path = dataset_dir / f"tuning-summary-{dataset}.csv"
    nested_tuning_path = dataset_dir / f"nested-tuning-summary-{dataset}.csv"
    if final_path.exists() and not skip_final:
        return final_path
    if nested_tuning_path.exists():
        return nested_tuning_path
    return tuning_path


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rank_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: metric_key(row, metric))


def metric_key(row: dict[str, Any], metric: str) -> tuple[int, float]:
    value = parse_float(row.get(metric))
    if not math.isfinite(value):
        return (1, float("inf"))
    return (0, -value) if metric in MAXIMIZE_METRICS else (0, value)


def default_summary_metric(meta: dict[str, Any], final_summary: bool) -> str:
    if meta.get("task_type") == "classification":
        return "test_rocauc_mean" if final_summary else "best_val_rocauc_mean"
    return "test_mae_mean" if final_summary else "best_val_mae_mean"


def metric_from_rows(rows: list[dict[str, Any]], meta: dict[str, Any], final_summary: bool) -> str:
    if rows:
        selection = rows[0].get("selection_metric")
        if isinstance(selection, str) and selection and selection != "auto":
            if final_summary and selection.startswith("best_val_"):
                candidate = "test_" + selection[len("best_val_") :] + "_mean"
            else:
                candidate = f"{selection}_mean"
            if any(candidate in row for row in rows):
                return candidate
    return default_summary_metric(meta, final_summary)


def format_metric(mean_value: Any, std_value: Any = None) -> str:
    mean_float = parse_float(mean_value)
    if not math.isfinite(mean_float):
        return "n/a"
    if std_value is None:
        return f"{mean_float:.6g}"
    std_float = parse_float(std_value)
    if not math.isfinite(std_float):
        return f"{mean_float:.6g}"
    return f"{mean_float:.6g} +/- {std_float:.3g}"


def format_params(value: Any) -> str:
    number = parse_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{int(round(number)):,}"


def parameter_value(row: dict[str, Any], kind: str) -> Any:
    if kind == "before":
        return row.get("params_before_prune_mean", row.get("params_mean"))
    if kind == "after":
        return row.get("params_after_prune_mean", row.get("effective_params_mean", row.get("params_mean")))
    if kind == "pruned":
        return row.get("params_pruned_mean", row.get("pruned_params_mean", 0))
    if kind == "pruned_pct":
        if "params_pruned_pct_mean" in row:
            return row["params_pruned_pct_mean"]
        before = parse_float(parameter_value(row, "before"))
        pruned = parse_float(parameter_value(row, "pruned"))
        return 100.0 * pruned / before if math.isfinite(before) and before else 0.0
    raise ValueError(f"unknown parameter kind {kind!r}")


def add_parameter_aliases(row: dict[str, Any]) -> None:
    row.setdefault("params_before_prune_mean", parameter_value(row, "before"))
    row.setdefault("params_after_prune_mean", parameter_value(row, "after"))
    row.setdefault("params_pruned_mean", parameter_value(row, "pruned"))
    row.setdefault("params_pruned_pct_mean", parameter_value(row, "pruned_pct"))
    if "mlp_effective_params_budget" in row and "mlp_params_after_prune_budget" not in row:
        row["mlp_params_after_prune_budget"] = row["mlp_effective_params_budget"]


def print_dataset_summary(
    dataset: str,
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
    summary_path: Path,
    final_summary: bool,
) -> list[dict[str, Any]]:
    benchmark_rows = [
        row
        for row in rows
        if row.get("evaluation_variant", "unpruned-benchmark") == "unpruned-benchmark"
    ]
    if benchmark_rows:
        rows = benchmark_rows
    metric = metric_from_rows(rows, meta, final_summary)
    ranked = rank_rows(rows, metric)
    unit = meta.get("unit")
    unit_text = f" [{unit}]" if unit not in (None, "", "None") else ""

    print(f"\n### {dataset}", flush=True)
    print(
        f"target={meta['target']}{unit_text} | task={meta['task_type']} | input={meta['input_type']} | "
        f"n={int(meta['n_samples']):,} | summary={summary_path.name}",
        flush=True,
    )
    if not ranked:
        print("No summary rows found yet.", flush=True)
        return []

    table_rows = []
    if meta.get("task_type") == "classification":
        headers = [
            "rank",
            "model",
            "rocauc",
            "acc",
            "f1",
            "val_rocauc",
            "params_before",
            "params_after",
            "pruned",
            "pruned_pct",
            "mlp_budget",
            "budget_ok",
            "trial_id",
        ]
    else:
        headers = [
            "rank",
            "model",
            "mae",
            "rmse",
            "r2",
            "val_mae",
            "params_before",
            "params_after",
            "pruned",
            "pruned_pct",
            "mlp_budget",
            "budget_ok",
            "trial_id",
        ]
    for rank, row in enumerate(ranked, start=1):
        add_parameter_aliases(row)
        base = {
            "rank": rank,
            "model": row.get("model_family") or row.get("model") or "",
            "params_before": format_params(parameter_value(row, "before")),
            "params_after": format_params(parameter_value(row, "after")),
            "pruned": format_params(parameter_value(row, "pruned")),
            "pruned_pct": format_metric(parameter_value(row, "pruned_pct")),
            "mlp_budget": format_params(row.get("mlp_params_after_prune_budget", row.get("mlp_effective_params_budget"))),
            "budget_ok": row.get("parameter_budget_ok", ""),
            "trial_id": row.get("trial_id", ""),
        }
        if meta.get("task_type") == "classification":
            base.update(
                {
                    "rocauc": format_metric(row.get("test_rocauc_mean"), row.get("test_rocauc_std")),
                    "acc": format_metric(row.get("test_accuracy_mean"), row.get("test_accuracy_std")),
                    "f1": format_metric(row.get("test_f1_mean"), row.get("test_f1_std")),
                    "val_rocauc": format_metric(row.get("best_val_rocauc_mean"), row.get("best_val_rocauc_std")),
                }
            )
        else:
            base.update(
                {
                    "mae": format_metric(row.get("test_mae_mean"), row.get("test_mae_std")),
                    "rmse": format_metric(row.get("test_rmse_mean"), row.get("test_rmse_std")),
                    "r2": format_metric(row.get("test_r2_mean"), row.get("test_r2_std")),
                    "val_mae": format_metric(row.get("best_val_mae_mean"), row.get("best_val_mae_std")),
                }
            )
        table_rows.append(base)
    print_table(table_rows, headers)

    best = ranked[0]
    best_model = best.get("model_family") or best.get("model")
    best_score = parse_float(best.get(metric))
    if math.isfinite(best_score):
        print(f"Winner: {best_model} ({metric}={best_score:.6g})", flush=True)
    else:
        print(f"Winner: {best_model} (metric unavailable)", flush=True)

    aggregate_rows = []
    for rank, row in enumerate(ranked, start=1):
        aggregate_rows.append(
            {
                **meta,
                **row,
                "rank": rank,
                "summary_metric": metric,
                "summary_path": str(summary_path),
            }
        )
    return aggregate_rows


def print_table(rows: list[dict[str, Any]], headers: list[str]) -> None:
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers), flush=True)
    print("-+-".join("-" * widths[header] for header in headers), flush=True)
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers), flush=True)


def write_outputs(
    output_root: Path,
    aggregate_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "all-datasets-summary.csv"
    for row in aggregate_rows:
        add_parameter_aliases(row)
    if aggregate_rows:
        fieldnames = ordered_fieldnames(aggregate_rows)
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(aggregate_rows)

    payload = {
        "args": vars(args),
        "supported_datasets": [
            metadata_row(dataset, metadata_for_dataset(dataset, metadata))
            for dataset in supported_datasets(metadata, args)
        ],
        "aggregate_rows": aggregate_rows,
        "failures": failures,
    }
    json_path = output_root / "all-datasets-summary.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if aggregate_rows:
        print(f"\nUpdated aggregate summary: {csv_path}", flush=True)
    print(f"Updated aggregate JSON: {json_path}", flush=True)


def ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "dataset",
        "task_type",
        "input_type",
        "target",
        "unit",
        "n_samples",
        "rank",
        "summary_metric",
        "model",
        "model_family",
        "trial_id",
        "selection_metric",
        "folds",
        "prune_kan_fraction",
        "params_before_prune_mean",
        "params_after_prune_mean",
        "params_after_prune_std",
        "params_pruned_mean",
        "params_pruned_pct_mean",
        "mlp_params_after_prune_budget",
        "parameter_budget_ok",
        "formula_path",
        "formula_paths",
        "params_mean",
        "effective_params_mean",
        "effective_params_std",
        "pruned_params_mean",
        "test_mae_mean",
        "test_mae_std",
        "test_rmse_mean",
        "test_rmse_std",
        "test_r2_mean",
        "test_r2_std",
        "test_rocauc_mean",
        "test_rocauc_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_balanced_accuracy_mean",
        "test_balanced_accuracy_std",
        "test_f1_mean",
        "test_f1_std",
        "best_val_mae_mean",
        "best_val_mae_std",
        "best_val_rmse_mean",
        "best_val_rmse_std",
        "best_val_r2_mean",
        "best_val_r2_std",
        "best_val_rocauc_mean",
        "best_val_rocauc_std",
        "best_val_accuracy_mean",
        "best_val_accuracy_std",
        "best_val_balanced_accuracy_mean",
        "best_val_balanced_accuracy_std",
        "best_val_f1_mean",
        "best_val_f1_std",
        "n_features",
        "common_dim",
        "group_dim",
        "property_dim",
        "target_dim",
        "kan_grid_size",
        "kan_spline_order",
        "lr",
        "weight_decay",
        "dropout",
        "loss",
        "train_seconds_mean",
        "forward_ms_per_batch_mean",
        "summary_path",
    ]
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def existing_summary(dataset_dir: Path, dataset: str, skip_final: bool) -> Path | None:
    path = summary_csv_path(dataset_dir, dataset, skip_final)
    return path if path.exists() else None


def main() -> None:
    args = parse_args()
    metadata = matbench_metadata()
    if args.list_datasets:
        for dataset in supported_datasets(metadata, args):
            row = metadata_row(dataset, metadata_for_dataset(dataset, metadata))
            unit = row["unit"] if row["unit"] not in (None, "None") else ""
            frac_true = (
                f", frac_true={row['frac_true']:.3g}"
                if row["task_type"] == "classification" and row.get("frac_true") is not None
                else ""
            )
            print(
                f"{dataset}: {row['input_type']} {row['task_type']}, target={row['target']} {unit}, "
                f"n={row['n_samples']:,}{frac_true}",
                flush=True,
            )
        return

    datasets, skipped = select_datasets(args, metadata)
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "benchmarks" / f"tune-modnet-all-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Output root: {output_root}", flush=True)
    print(f"Datasets: {', '.join(datasets)}", flush=True)
    if skipped:
        print(f"Skipped by request: {', '.join(skipped)}", flush=True)
    if not args.evaluate_tune_test:
        print("Protocol: tune on train/validation only; final summaries use official Matbench holdout folds.", flush=True)
    if not args.allow_kan_larger_than_mlp:
        print("KAN budget: final KAN selection prefers effective_params below the selected MLP.", flush=True)

    aggregate_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets, start=1):
        dataset_dir = output_root / dataset
        dataset_meta = metadata_row(dataset, metadata_for_dataset(dataset, metadata))
        print(f"\n=== Dataset {index}/{len(datasets)}: {dataset} ===", flush=True)
        cmd = build_tune_command(args, dataset, dataset_dir)
        print(subprocess.list2cmdline(cmd), flush=True)

        summary_path = existing_summary(dataset_dir, dataset, args.skip_final) if args.resume else None
        try:
            if summary_path is not None:
                print(f"Resume: using existing {summary_path}", flush=True)
            elif args.dry_run:
                print("Dry run: command not executed.", flush=True)
                continue
            else:
                subprocess.run(cmd, check=True, cwd=ROOT)
                summary_path = summary_csv_path(dataset_dir, dataset, args.skip_final)

            rows = read_csv_rows(summary_path) if summary_path else []
            final_summary = summary_path is not None and summary_path.name.startswith("final-summary")
            aggregate_rows.extend(
                print_dataset_summary(dataset, dataset_meta, rows, summary_path or dataset_dir, final_summary)
            )
        except Exception as exc:
            failures.append(
                {
                    "dataset": dataset,
                    "error": str(exc),
                    "output_dir": str(dataset_dir),
                }
            )
            print(f"FAILED {dataset}: {exc}", flush=True)
            if args.fail_fast:
                raise
        finally:
            write_outputs(output_root, aggregate_rows, failures, metadata, args)

    if failures:
        print("\nFailures:", flush=True)
        for failure in failures:
            print(f"- {failure['dataset']}: {failure['error']}", flush=True)
    else:
        print("\nAll requested datasets completed.", flush=True)


if __name__ == "__main__":
    main()
