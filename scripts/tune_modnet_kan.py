from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PRESETS = [
    "auto",
    "pymatgen-composition",
    "pymatgen-structure",
    "matminer-composition",
    "matminer-structure-lite",
]
MODEL_FAMILIES = ["mlp", "fastkan", "spline"]
TASK_TYPES = ("regression", "classification")
METRIC_NAMES = [
    "mae",
    "rmse",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "rocauc",
]
MAXIMIZE_METRICS = {
    "best_val_accuracy",
    "best_val_balanced_accuracy",
    "best_val_f1",
    "best_val_rocauc",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_f1",
    "test_rocauc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune MODNet-style MLP, FastKAN, and B-spline KAN hyperparameters, "
            "then run Matbench-aligned final benchmarks."
        )
    )
    parser.add_argument("--dataset", default="matbench_phonons")
    parser.add_argument("--model-families", nargs="+", choices=MODEL_FAMILIES, default=MODEL_FAMILIES)
    parser.add_argument("--tune-folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--final-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--search-space", choices=["compact", "random", "grid"], default="compact")
    parser.add_argument(
        "--num-random-trials",
        type=int,
        default=8,
        help="Number of random trials per model family.",
    )
    parser.add_argument("--max-trials-per-family", type=int, default=None)
    parser.add_argument(
        "--metric",
        choices=[
            "auto",
            "best_val_mae",
            "best_val_rmse",
            "best_val_accuracy",
            "best_val_balanced_accuracy",
            "best_val_f1",
            "best_val_rocauc",
            "test_mae",
            "test_rmse",
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
    parser.add_argument("--kan-grid-size-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--kan-spline-order-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--lr-candidates", type=float, nargs="+", default=None)
    parser.add_argument("--weight-decay-candidates", type=float, nargs="+", default=None)
    parser.add_argument("--dropout-candidates", type=float, nargs="+", default=None)
    parser.add_argument(
        "--loss-candidates",
        nargs="+",
        choices=["mae", "rmse", "mse", "bce"],
        default=["mae", "rmse"],
    )
    parser.add_argument("--prune-kan-fraction-candidates", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--allow-kan-larger-than-mlp",
        action="store_true",
        help="Do not filter KAN candidates by the selected MLP parameter budget.",
    )
    parser.add_argument("--scaler", choices=["minmax", "standard", "none"], default="minmax")
    parser.add_argument("--target-scale", choices=["none", "standard"], default="none")
    parser.add_argument("--impute-strategy", default="median")
    parser.add_argument("--tune-epochs", type=int, default=80)
    parser.add_argument("--final-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=60)
    parser.add_argument("--tune-train-size", type=int, default=512)
    parser.add_argument(
        "--tune-test-size",
        type=int,
        default=256,
        help="Diagnostic test subset size during tuning; only used with --evaluate-tune-test.",
    )
    parser.add_argument(
        "--evaluate-tune-test",
        action="store_true",
        help=(
            "Also evaluate the official Matbench holdout fold during tuning. "
            "Default is off so hyperparameter selection uses only train/validation data."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--log-every-epochs", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-export-final-formulas",
        action="store_true",
        help="Do not write sparse formula text files for final KAN-family benchmarks.",
    )
    parser.add_argument("--formula-top-k", type=int, default=40)
    parser.add_argument("--formula-min-abs", type=float, default=0.0)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def matbench_task_type(dataset: str) -> str:
    from matbench.metadata import mbv01_metadata

    return str(mbv01_metadata[dataset].task_type)


def resolve_metric(metric: str, task_type: str) -> str:
    if metric != "auto":
        return metric
    return "best_val_rocauc" if task_type == "classification" else "best_val_mae"


def resolve_loss_candidates(losses: list[str], task_type: str) -> list[str]:
    if task_type == "classification":
        return ["bce"]

    resolved = []
    for loss in losses:
        loss = "rmse" if loss == "mse" else loss
        if loss == "bce":
            raise ValueError("Regression tasks cannot use --loss-candidates bce")
        if loss not in resolved:
            resolved.append(loss)
    return resolved


def metric_sort_value(value: Any, metric: str) -> tuple[int, float]:
    number = safe_float(value)
    if not math.isfinite(number):
        return (1, float("inf"))
    if metric in MAXIMIZE_METRICS:
        return (0, -number)
    return (0, number)


def candidate_space(args: argparse.Namespace) -> dict[str, list[Any]]:
    return {
        "n_features": args.n_feature_candidates or [128, 256, 280, 512],
        "common_dim": args.common_dim_candidates or [64, 128, 512],
        "group_dim": args.group_dim_candidates or [32, 64, 128],
        "property_dim": args.property_dim_candidates or [16, 32, 64],
        "kan_grid_size": args.kan_grid_size_candidates or [3, 5],
        "kan_spline_order": args.kan_spline_order_candidates or [3],
        "lr": args.lr_candidates or [3e-4, 1e-3, 0.005],
        "weight_decay": args.weight_decay_candidates or [0.0, 1e-6, 1e-5],
        "dropout": args.dropout_candidates or [0.0, 0.05],
        "loss": args.loss_candidates,
        "prune_kan_fraction": args.prune_kan_fraction_candidates or [0.0],
    }


def make_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    trials = []
    for family in args.model_families:
        if args.search_space == "grid":
            family_trials = grid_trials(args, family)
        elif args.search_space == "random":
            family_trials = random_trials(args, family)
        else:
            family_trials = compact_trials(args, family)
        if args.max_trials_per_family is not None:
            family_trials = family_trials[: args.max_trials_per_family]
        trials.extend(family_trials)
    return trials


def compact_trials(args: argparse.Namespace, family: str) -> list[dict[str, Any]]:
    space = candidate_space(args)
    base_specs = [
        (space["n_features"][0], space["common_dim"][0], space["group_dim"][0], space["property_dim"][0], 3, 3, space["lr"][0], space["weight_decay"][0], space["dropout"][0]),
        (space["n_features"][min(1, len(space["n_features"]) - 1)], space["common_dim"][0], space["group_dim"][0], space["property_dim"][0], 5, 3, space["lr"][min(1, len(space["lr"]) - 1)], space["weight_decay"][0], space["dropout"][0]),
        (space["n_features"][min(2, len(space["n_features"]) - 1)], space["common_dim"][min(1, len(space["common_dim"]) - 1)], space["group_dim"][min(1, len(space["group_dim"]) - 1)], space["property_dim"][min(1, len(space["property_dim"]) - 1)], 3, 3, space["lr"][min(1, len(space["lr"]) - 1)], space["weight_decay"][min(1, len(space["weight_decay"]) - 1)], space["dropout"][min(1, len(space["dropout"]) - 1)]),
        (280 if 280 in space["n_features"] else space["n_features"][min(2, len(space["n_features"]) - 1)], 512 if 512 in space["common_dim"] else space["common_dim"][-1], 128 if 128 in space["group_dim"] else space["group_dim"][-1], 64 if 64 in space["property_dim"] else space["property_dim"][-1], 5, 3, 0.005 if 0.005 in space["lr"] else space["lr"][-1], 0.0, 0.0),
    ]
    trials: dict[str, dict[str, Any]] = {}
    prune_values = [0.0] if family == "mlp" else space["prune_kan_fraction"]
    for spec in base_specs:
        for loss in space["loss"]:
            for prune_fraction in prune_values:
                trial = make_trial(family, *spec, loss, prune_fraction)
                trials[trial["trial_id"]] = trial
    return list(trials.values())


def random_trials(args: argparse.Namespace, family: str) -> list[dict[str, Any]]:
    space = candidate_space(args)
    rng = random.Random(args.seed + MODEL_FAMILIES.index(family))
    trials: dict[str, dict[str, Any]] = {}
    for trial in compact_trials(args, family):
        trials[trial["trial_id"]] = trial
        if len(trials) >= args.num_random_trials:
            return list(trials.values())

    max_attempts = max(100, args.num_random_trials * 20)
    for _ in range(max_attempts):
        trial = make_trial(
            family,
            rng.choice(space["n_features"]),
            rng.choice(space["common_dim"]),
            rng.choice(space["group_dim"]),
            rng.choice(space["property_dim"]),
            rng.choice(space["kan_grid_size"]),
            rng.choice(space["kan_spline_order"]),
            rng.choice(space["lr"]),
            rng.choice(space["weight_decay"]),
            rng.choice(space["dropout"]),
            rng.choice(space["loss"]),
            0.0 if family == "mlp" else rng.choice(space["prune_kan_fraction"]),
        )
        trials.setdefault(trial["trial_id"], trial)
        if len(trials) >= args.num_random_trials:
            break
    return list(trials.values())


def grid_trials(args: argparse.Namespace, family: str) -> list[dict[str, Any]]:
    space = candidate_space(args)
    grid_sizes = [0] if family == "mlp" else space["kan_grid_size"]
    spline_orders = [0] if family != "spline" else space["kan_spline_order"]
    return [
        make_trial(family, *spec)
        for spec in itertools.product(
            space["n_features"],
            space["common_dim"],
            space["group_dim"],
            space["property_dim"],
            grid_sizes,
            spline_orders,
            space["lr"],
            space["weight_decay"],
            space["dropout"],
            space["loss"],
            [0.0] if family == "mlp" else space["prune_kan_fraction"],
        )
    ]


def make_trial(
    family: str,
    n_features: int,
    common_dim: int,
    group_dim: int,
    property_dim: int,
    kan_grid_size: int,
    kan_spline_order: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    loss: str,
    prune_kan_fraction: float,
) -> dict[str, Any]:
    grid_part = "mlp" if family == "mlp" else f"kg{kan_grid_size}"
    if family == "spline":
        grid_part += f"_ko{kan_spline_order}"
    prune_part = "" if family == "mlp" or prune_kan_fraction <= 0 else f"_prune{format_float_id(prune_kan_fraction)}"
    trial_id = (
        f"{family}_nf{n_features}_c{common_dim}_g{group_dim}_p{property_dim}_"
        f"{grid_part}_lr{format_float_id(lr)}_"
        f"wd{format_float_id(weight_decay)}_do{format_float_id(dropout)}_loss{loss}{prune_part}"
    )
    return {
        "trial_id": trial_id,
        "model_family": family,
        "n_features": int(n_features),
        "common_dim": int(common_dim),
        "group_dim": int(group_dim),
        "property_dim": int(property_dim),
        "kan_grid_size": int(kan_grid_size),
        "kan_spline_order": int(kan_spline_order),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "loss": loss,
        "prune_kan_fraction": 0.0 if family == "mlp" else float(prune_kan_fraction),
    }


def format_float_id(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def benchmark_command(
    args: argparse.Namespace,
    trial: dict[str, Any],
    folds: list[int],
    epochs: int,
    output_dir: Path,
    train_size: int | None,
    test_size: int | None,
    tuning_mode: bool,
    export_formulas: bool,
) -> list[str]:
    family = trial["model_family"]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_modnet_kan.py"),
        "--dataset",
        args.dataset,
        "--folds",
        *[str(fold) for fold in folds],
        "--models",
        family,
        "--featurizer-preset",
        args.featurizer_preset,
        "--featurizer-jobs",
        str(args.featurizer_jobs),
        "--n-features",
        str(trial["n_features"]),
        "--common-dims",
        str(trial["common_dim"]),
        "--group-dims",
        str(trial["group_dim"]),
        "--property-dims",
        str(trial["property_dim"]),
        "--kan-grid-size",
        str(max(1, int(trial["kan_grid_size"]))),
        "--kan-spline-order",
        str(max(1, int(trial["kan_spline_order"]))),
        "--scaler",
        args.scaler,
        "--target-scale",
        args.target_scale,
        "--impute-strategy",
        args.impute_strategy,
        "--dropout",
        str(trial["dropout"]),
        "--loss",
        str(trial["loss"]),
        "--prune-kan-fraction",
        str(trial.get("prune_kan_fraction", 0.0)),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--val-ratio",
        str(args.val_ratio),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--lr",
        str(trial["lr"]),
        "--weight-decay",
        str(trial["weight_decay"]),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--log-every-epochs",
        str(args.log_every_epochs),
        "--output-dir",
        str(output_dir),
    ]
    if args.require_cuda:
        cmd.append("--require-cuda")
    if train_size is not None:
        cmd.extend(["--train-size", str(train_size)])
    if test_size is not None and (not tuning_mode or args.evaluate_tune_test):
        cmd.extend(["--test-size", str(test_size)])
    if tuning_mode:
        cmd.extend(["--forward-iters", "1", "--warmup-iters", "0", "--no-matbench-records"])
        if not args.evaluate_tune_test:
            cmd.append("--skip-test-eval")
    if export_formulas:
        cmd.extend(
            [
                "--export-formulas",
                "--formula-top-k",
                str(args.formula_top_k),
                "--formula-min-abs",
                str(args.formula_min_abs),
            ]
        )
    return cmd


def run_benchmark(cmd: list[str], output_dir: Path, dataset: str) -> dict[str, Any]:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    paths = sorted(output_dir.glob(f"modnet-kan-{dataset}-*.json"))
    if not paths:
        raise FileNotFoundError(f"No benchmark JSON found in {output_dir}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def summarize_trial(
    trial: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    family = trial["model_family"]
    for fold_payload in payload["folds"]:
        for result in fold_payload["results"]:
            if result["model"] != family:
                continue
            rows.append(
                {
                    **trial,
                    "fold": int(result["fold"]),
                    "task_type": result.get("task_type", ""),
                    "params": int(result["params"]),
                    "effective_params": int(result.get("effective_params", result["params"])),
                    "pruned_params": int(result.get("pruned_params", 0)),
                    "params_before_prune": int(result.get("params_before_prune", result["params"])),
                    "params_after_prune": int(result.get("params_after_prune", result.get("effective_params", result["params"]))),
                    "params_pruned": int(result.get("params_pruned", result.get("pruned_params", 0))),
                    "params_pruned_pct": safe_float(result.get("params_pruned_pct", 0.0)),
                    "train_seconds": safe_float(result["train_seconds"]),
                    "forward_ms_per_batch": safe_float(result["forward_ms_per_batch"]),
                    **{
                        f"{prefix}_{metric}": safe_float(result.get(f"{prefix}_{metric}"))
                        for prefix in ("best_val", "test")
                        for metric in METRIC_NAMES
                    },
                }
            )

    summary = {
        **trial,
        "folds": len(rows),
        "params_mean": mean(float(row["params"]) for row in rows),
        "effective_params_mean": mean(float(row["effective_params"]) for row in rows),
        "effective_params_std": stdev(float(row["effective_params"]) for row in rows) if len(rows) > 1 else 0.0,
        "pruned_params_mean": mean(float(row["pruned_params"]) for row in rows),
        "params_before_prune_mean": mean(float(row["params_before_prune"]) for row in rows),
        "params_after_prune_mean": mean(float(row["params_after_prune"]) for row in rows),
        "params_after_prune_std": stdev(float(row["params_after_prune"]) for row in rows) if len(rows) > 1 else 0.0,
        "params_pruned_mean": mean(float(row["params_pruned"]) for row in rows),
        "params_pruned_pct_mean": mean(float(row["params_pruned_pct"]) for row in rows),
    }
    for metric in (
        *[f"{prefix}_{name}" for prefix in ("best_val", "test") for name in METRIC_NAMES],
        "train_seconds",
        "forward_ms_per_batch",
    ):
        values = [safe_float(row[metric]) for row in rows]
        finite_values = [value for value in values if math.isfinite(value)]
        summary[f"{metric}_mean"] = mean(finite_values) if finite_values else float("nan")
        summary[f"{metric}_std"] = stdev(finite_values) if len(finite_values) > 1 else 0.0
    return rows, summary


def best_trials_by_family(
    summary_rows: list[dict[str, Any]],
    metric: str,
    enforce_kan_budget: bool,
) -> dict[str, dict[str, Any]]:
    metric_mean = f"{metric}_mean"
    best = {}
    mlp_rows = [row for row in summary_rows if row["model_family"] == "mlp"]
    mlp_budget = None
    if mlp_rows:
        best_mlp = min(mlp_rows, key=lambda row: metric_sort_value(row.get(metric_mean), metric))
        mlp_budget = safe_float(best_mlp.get("effective_params_mean", best_mlp.get("params_mean")))

    for family in MODEL_FAMILIES:
        rows = [row for row in summary_rows if row["model_family"] == family]
        if rows:
            budgeted_rows = rows
            if (
                enforce_kan_budget
                and family != "mlp"
                and mlp_budget is not None
                and math.isfinite(mlp_budget)
            ):
                budgeted_rows = [
                    row
                    for row in rows
                    if safe_float(row.get("effective_params_mean", row.get("params_mean"))) < mlp_budget
                ]
                if not budgeted_rows:
                    print(
                        f"Skipping final selection for {family}: no tuning trial has "
                        f"effective_params_mean below the selected MLP budget ({mlp_budget:.0f}). "
                        "Use --allow-kan-larger-than-mlp for ablations.",
                        flush=True,
                    )
                    continue
            best[family] = min(budgeted_rows, key=lambda row: metric_sort_value(row.get(metric_mean), metric))
            if family != "mlp" and mlp_budget is not None:
                best[family]["mlp_effective_params_budget"] = mlp_budget
                best[family]["mlp_params_after_prune_budget"] = mlp_budget
                best[family]["parameter_budget_ok"] = (
                    safe_float(best[family].get("effective_params_mean", best[family].get("params_mean"))) < mlp_budget
                )
    return best


def summarize_final_payload(
    trial: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    family = trial["model_family"]
    fold_rows = []
    for fold_payload in payload["folds"]:
        for result in fold_payload["results"]:
            if result["model"] != family:
                continue
            fold_rows.append({**trial, **result})

    summary_rows = [{**trial, **summary} for summary in payload.get("summary", []) if summary["model"] == family]
    matbench_record = payload.get("matbench_records", {}).get(family, {})
    return fold_rows, summary_rows, matbench_record


def safe_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    number = float(value)
    return number


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: list[dict[str, Any]], metric: str) -> None:
    metric_mean = f"{metric}_mean"
    headers = [
        "rank",
        "model_family",
        "trial_id",
        "folds",
        "prune_kan_fraction",
        "params_before_prune_mean",
        "params_after_prune_mean",
        "params_pruned_mean",
        "params_pruned_pct_mean",
        "mlp_params_after_prune_budget",
        "parameter_budget_ok",
        metric_mean,
    ]
    for optional in ("best_val_mae_mean", "best_val_rmse_mean", "best_val_rocauc_mean", "test_mae_mean", "test_rmse_mean", "test_rocauc_mean"):
        if optional != metric_mean and any(optional in row for row in summary_rows):
            headers.append(optional)
    widths = {
        header: max(
            len(header),
            *[
                len(format_value(idx + 1 if header == "rank" else row.get(header, "")))
                for idx, row in enumerate(summary_rows)
            ],
        )
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for idx, row in enumerate(summary_rows, start=1):
        values = {"rank": idx, **row}
        print(" | ".join(format_value(values.get(header, "")).ljust(widths[header]) for header in headers))
    print(f"\nSelection metric: {metric_mean}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def add_parameter_aliases(row: dict[str, Any]) -> None:
    if "params_before_prune" not in row and "params" in row:
        row["params_before_prune"] = row["params"]
    if "params_after_prune" not in row:
        row["params_after_prune"] = row.get("effective_params", row.get("params"))
    if "params_pruned" not in row:
        row["params_pruned"] = row.get("pruned_params", 0)
    if "params_pruned_pct" not in row:
        before = safe_float(row.get("params_before_prune"))
        pruned = safe_float(row.get("params_pruned"))
        row["params_pruned_pct"] = 100.0 * pruned / before if math.isfinite(before) and before else 0.0

    if "params_before_prune_mean" not in row and "params_mean" in row:
        row["params_before_prune_mean"] = row["params_mean"]
    if "params_after_prune_mean" not in row:
        row["params_after_prune_mean"] = row.get("effective_params_mean", row.get("params_mean"))
    if "params_after_prune_std" not in row and "effective_params_std" in row:
        row["params_after_prune_std"] = row["effective_params_std"]
    if "params_pruned_mean" not in row:
        row["params_pruned_mean"] = row.get("pruned_params_mean", 0)
    if "params_pruned_pct_mean" not in row:
        before = safe_float(row.get("params_before_prune_mean"))
        pruned = safe_float(row.get("params_pruned_mean"))
        row["params_pruned_pct_mean"] = 100.0 * pruned / before if math.isfinite(before) and before else 0.0


def ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "dataset",
        "task_type",
        "selection_metric",
        "rank",
        "model",
        "model_family",
        "trial_id",
        "fold",
        "folds",
        "prune_kan_fraction",
        "params_before_prune",
        "params_after_prune",
        "params_pruned",
        "params_pruned_pct",
        "params_before_prune_mean",
        "params_after_prune_mean",
        "params_after_prune_std",
        "params_pruned_mean",
        "params_pruned_pct_mean",
        "mlp_params_after_prune_budget",
        "parameter_budget_ok",
        "formula_path",
        "formula_paths",
        "params",
        "effective_params",
        "pruned_params",
        "params_mean",
        "effective_params_mean",
        "effective_params_std",
        "pruned_params_mean",
        "best_val_mae",
        "best_val_rmse",
        "best_val_rocauc",
        "best_val_accuracy",
        "best_val_balanced_accuracy",
        "best_val_f1",
        "test_mae",
        "test_rmse",
        "test_rocauc",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "best_val_mae_mean",
        "best_val_rmse_mean",
        "best_val_rocauc_mean",
        "best_val_accuracy_mean",
        "best_val_balanced_accuracy_mean",
        "best_val_f1_mean",
        "test_mae_mean",
        "test_rmse_mean",
        "test_rocauc_mean",
        "test_accuracy_mean",
        "test_balanced_accuracy_mean",
        "test_f1_mean",
        "n_features",
        "common_dim",
        "group_dim",
        "property_dim",
        "kan_grid_size",
        "kan_spline_order",
        "lr",
        "weight_decay",
        "dropout",
        "loss",
        "train_seconds",
        "train_seconds_mean",
        "forward_ms_per_batch",
        "forward_ms_per_batch_mean",
    ]
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def main() -> None:
    args = parse_args()
    task_type = matbench_task_type(args.dataset)
    if task_type not in TASK_TYPES:
        raise ValueError(f"{args.dataset} task_type is {task_type!r}; expected one of {TASK_TYPES}")
    args.metric = resolve_metric(args.metric, task_type)
    args.loss_candidates = resolve_loss_candidates(args.loss_candidates, task_type)
    if task_type == "classification":
        args.target_scale = "none"
    if args.val_ratio <= 0 and args.metric.startswith("best_val_"):
        raise ValueError(f"--metric {args.metric} requires --val-ratio > 0")
    if args.metric.startswith("test_") and not args.evaluate_tune_test:
        raise ValueError(
            f"--metric {args.metric} requires --evaluate-tune-test, but Matbench-strict "
            "selection should normally use a best_val_* metric."
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "benchmarks" / f"tune-modnet-family-{args.dataset}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    trials = make_trials(args)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Model families: {args.model_families}", flush=True)
    print(f"Tune folds: {args.tune_folds}", flush=True)
    print(f"Final folds: {args.final_folds}", flush=True)
    print(f"Trials: {len(trials)}", flush=True)
    print(f"Metric: {args.metric}", flush=True)
    print(f"Task type: {task_type}", flush=True)
    print(f"Loss candidates: {args.loss_candidates}", flush=True)
    if not args.evaluate_tune_test:
        print("Tuning phase: official Matbench test folds are skipped.", flush=True)

    if args.dry_run:
        print(json.dumps({"trials": trials}, indent=2), flush=True)
        return

    fold_rows = []
    summary_rows = []
    for trial in trials:
        trial_dir = output_dir / "tuning" / trial["model_family"] / trial["trial_id"]
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Trial {trial['trial_id']} ===", flush=True)
        cmd = benchmark_command(
            args,
            trial,
            folds=args.tune_folds,
            epochs=args.tune_epochs,
            output_dir=trial_dir,
            train_size=args.tune_train_size,
            test_size=args.tune_test_size,
            tuning_mode=True,
            export_formulas=False,
        )
        payload = run_benchmark(cmd, trial_dir, args.dataset)
        rows, summary = summarize_trial(trial, payload)
        for row in rows:
            row["selection_metric"] = args.metric
        summary["selection_metric"] = args.metric
        summary["task_type"] = task_type
        fold_rows.extend(rows)
        summary_rows.append(summary)

    metric_mean = f"{args.metric}_mean"
    summary_rows.sort(key=lambda row: (str(row["model_family"]), metric_sort_value(row.get(metric_mean), args.metric)))
    ranked_rows = sorted(summary_rows, key=lambda row: metric_sort_value(row.get(metric_mean), args.metric))
    print()
    print_summary(ranked_rows, args.metric)

    fold_csv = output_dir / f"tuning-fold-results-{args.dataset}.csv"
    summary_csv = output_dir / f"tuning-summary-{args.dataset}.csv"
    write_csv(fold_csv, fold_rows)
    write_csv(summary_csv, summary_rows)

    best_by_family = best_trials_by_family(
        summary_rows,
        args.metric,
        enforce_kan_budget=not args.allow_kan_larger_than_mlp,
    )
    final_fold_rows = []
    final_summary_rows = []
    final_records = {}
    final_payloads = {}
    final_commands = {}
    final_dir = output_dir / "final-benchmark"
    if not args.skip_final:
        final_dir.mkdir(parents=True, exist_ok=True)
        for family, best in best_by_family.items():
            family_dir = final_dir / family
            family_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== Final benchmark: {family} / {best['trial_id']} ===", flush=True)
            cmd = benchmark_command(
                args,
                best,
                folds=args.final_folds,
                epochs=args.final_epochs,
                output_dir=family_dir,
                train_size=None,
                test_size=None,
                tuning_mode=False,
                export_formulas=family != "mlp" and not args.no_export_final_formulas,
            )
            final_commands[family] = cmd
            payload = run_benchmark(cmd, family_dir, args.dataset)
            final_payloads[family] = payload
            rows, summaries, matbench_record = summarize_final_payload(best, payload)
            for row in rows:
                row["selection_metric"] = args.metric
                row["task_type"] = task_type
                add_parameter_aliases(row)
            for summary in summaries:
                summary["selection_metric"] = args.metric
                summary["task_type"] = task_type
                add_parameter_aliases(summary)
                if family != "mlp" and "mlp_effective_params_budget" in best:
                    summary["mlp_effective_params_budget"] = best["mlp_effective_params_budget"]
                    summary["mlp_params_after_prune_budget"] = best["mlp_effective_params_budget"]
                    summary["parameter_budget_ok"] = (
                        safe_float(summary.get("effective_params_mean", summary.get("params_mean")))
                        < safe_float(best["mlp_effective_params_budget"])
                    )
            final_fold_rows.extend(rows)
            final_summary_rows.extend(summaries)
            final_records[family] = matbench_record

    final_fold_csv = output_dir / f"final-fold-results-{args.dataset}.csv"
    final_summary_csv = output_dir / f"final-summary-{args.dataset}.csv"
    write_csv(final_fold_csv, final_fold_rows)
    write_csv(final_summary_csv, final_summary_rows)

    best_payload = {
        "dataset": args.dataset,
        "task_type": task_type,
        "selection_metric": args.metric,
        "loss_candidates": args.loss_candidates,
        "prune_kan_fraction_candidates": args.prune_kan_fraction_candidates,
        "enforce_kan_smaller_than_mlp": not args.allow_kan_larger_than_mlp,
        "best_by_family": best_by_family,
        "best_overall": ranked_rows[0] if ranked_rows else None,
        "final_commands": final_commands,
        "final_summary": final_summary_rows,
        "final_matbench_records": final_records,
        "all_trials": summary_rows,
    }
    best_json = output_dir / "best_config.json"
    best_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    print("\nBest trials by family:")
    for family, row in best_by_family.items():
        budget_text = ""
        if family != "mlp" and "parameter_budget_ok" in row:
            budget_text = (
                f", effective_params={safe_float(row.get('effective_params_mean')):.0f}, "
                f"mlp_budget={safe_float(row.get('mlp_effective_params_budget')):.0f}, "
                f"budget_ok={row.get('parameter_budget_ok')}"
            )
        print(f"- {family}: {row['trial_id']} ({metric_mean}={safe_float(row.get(metric_mean)):.6g}{budget_text})")
    print(f"\nWrote {fold_csv}")
    print(f"Wrote {summary_csv}")
    if final_summary_rows:
        print(f"Wrote {final_fold_csv}")
        print(f"Wrote {final_summary_csv}")
    print(f"Wrote {best_json}")


if __name__ == "__main__":
    main()
