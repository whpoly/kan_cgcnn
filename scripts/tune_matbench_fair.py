from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
KAN_PROFILES = ["kan-readout", "kan-conv", "kan-full"]
METRICS = ["best_val_mae", "test_mae", "test_rmse", "train_seconds", "forward_ms_per_batch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and benchmark CGCNN, KAN readout, KAN conv, and full KAN profiles."
    )
    parser.add_argument("--dataset", default="matbench_phonons")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tune-profiles", nargs="+", choices=KAN_PROFILES, default=KAN_PROFILES)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dims", type=int, nargs="+", default=[32])
    parser.add_argument("--num-convs", type=int, default=4)
    parser.add_argument("--edge-dim", type=int, default=41)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--final-val-ratio", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--conv-kan-impl", choices=["fastkan", "spline"], default="fastkan")
    parser.add_argument("--head-kan-impl", choices=["fastkan", "spline"], default=None)
    parser.add_argument("--conv-kan-hidden-dim-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--conv-kan-grid-size-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--head-kan-hidden-dim-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--head-kan-grid-size-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--lrs", type=float, nargs="+", default=None)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=None)
    parser.add_argument("--dropouts", type=float, nargs="+", default=None)
    parser.add_argument("--search-space", choices=["random", "compact", "grid"], default="random")
    parser.add_argument("--num-random-trials", type=int, default=18)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--strategy", choices=["successive-halving", "full"], default="successive-halving")
    parser.add_argument("--halving-factor", type=int, default=3)
    parser.add_argument("--rung-epochs", type=int, nargs="+", default=None)
    parser.add_argument("--rung-fold-counts", type=int, nargs="+", default=None)
    parser.add_argument("--rung-train-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--metric", choices=["best_val_mae", "test_mae", "test_rmse"], default="best_val_mae")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=0)
    parser.add_argument("--epoch-pause-seconds", type=float, default=0.0)
    parser.add_argument("--forward-iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--run-final-benchmark", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def profile_settings(profile: str) -> dict[str, str]:
    if profile == "kan-readout":
        return {
            "comparison": "readout_only",
            "conv_net": "mlp",
            "mlp_head_net": "kan",
            "kan_head_net": "kan",
        }
    if profile == "kan-conv":
        return {
            "comparison": "message_only",
            "conv_net": "kan",
            "mlp_head_net": "mlp",
            "kan_head_net": "mlp",
        }
    if profile == "kan-full":
        return {
            "comparison": "full_model",
            "conv_net": "kan",
            "mlp_head_net": "mlp",
            "kan_head_net": "kan",
        }
    raise ValueError(f"unsupported profile {profile!r}")


def candidate_space(args: argparse.Namespace) -> dict[str, list[Any]]:
    return {
        "conv_hidden_dims": args.conv_kan_hidden_dim_candidates or [8, 16, 24],
        "conv_grid_sizes": args.conv_kan_grid_size_candidates or [2, 3, 4],
        "head_hidden_dims": args.head_kan_hidden_dim_candidates or [4, 8, 16, 32],
        "head_grid_sizes": args.head_kan_grid_size_candidates or [2, 3, 4],
        "lrs": args.lrs or [1e-3, 2e-3, 3e-3],
        "weight_decays": args.weight_decays or [0.0, 1e-5],
        "dropouts": args.dropouts or [0.0, 0.05],
    }


def make_trial(
    profile: str,
    conv_hidden_dim: int,
    conv_grid_size: int,
    head_hidden_dim: int,
    head_grid_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
) -> dict[str, Any]:
    settings = profile_settings(profile)
    trial_id = (
        f"{profile}_ch{conv_hidden_dim}_cg{conv_grid_size}_"
        f"rh{head_hidden_dim}_rg{head_grid_size}_"
        f"lr{format_float_id(lr)}_wd{format_float_id(weight_decay)}_"
        f"do{format_float_id(dropout)}"
    )
    return {
        "trial_id": trial_id,
        "profile": profile,
        **settings,
        "conv_kan_hidden_dim": conv_hidden_dim,
        "conv_kan_grid_size": conv_grid_size,
        "head_kan_hidden_dim": head_hidden_dim,
        "head_kan_grid_size": head_grid_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
    }


def compact_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    space = candidate_space(args)
    conv_hidden = space["conv_hidden_dims"]
    conv_grid = space["conv_grid_sizes"]
    head_hidden = space["head_hidden_dims"]
    head_grid = space["head_grid_sizes"]

    def pick(values: list[Any], index: int) -> Any:
        return values[min(index, len(values) - 1)]

    seeds = [
        ("kan-readout", pick(conv_hidden, 1), pick(conv_grid, 1), pick(head_hidden, 1), pick(head_grid, 1), 3e-3, 1e-5, 0.0),
        ("kan-readout", pick(conv_hidden, 1), pick(conv_grid, 1), pick(head_hidden, 2), pick(head_grid, 1), 1e-3, 1e-5, 0.0),
        ("kan-conv", pick(conv_hidden, 1), pick(conv_grid, 1), pick(head_hidden, 1), pick(head_grid, 1), 3e-3, 1e-5, 0.0),
        ("kan-conv", pick(conv_hidden, 2), pick(conv_grid, 1), pick(head_hidden, 1), pick(head_grid, 1), 1e-3, 1e-5, 0.0),
        ("kan-full", pick(conv_hidden, 1), pick(conv_grid, 1), pick(head_hidden, 1), pick(head_grid, 1), 3e-3, 1e-5, 0.0),
        ("kan-full", pick(conv_hidden, 1), pick(conv_grid, 2), pick(head_hidden, 2), pick(head_grid, 1), 1e-3, 1e-5, 0.0),
    ]
    return [
        make_trial(*seed)
        for seed in seeds
        if seed[0] in args.tune_profiles
    ]


def trial_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    return (
        trial["profile"],
        int(trial["conv_kan_hidden_dim"]),
        int(trial["conv_kan_grid_size"]),
        int(trial["head_kan_hidden_dim"]),
        int(trial["head_kan_grid_size"]),
        float(trial["lr"]),
        float(trial["weight_decay"]),
        float(trial["dropout"]),
    )


def random_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    space = candidate_space(args)
    rng = random.Random(args.seed)
    trials = {trial_key(trial): trial for trial in compact_trials(args)}
    max_combinations = (
        len(args.tune_profiles)
        * len(space["conv_hidden_dims"])
        * len(space["conv_grid_sizes"])
        * len(space["head_hidden_dims"])
        * len(space["head_grid_sizes"])
        * len(space["lrs"])
        * len(space["weight_decays"])
        * len(space["dropouts"])
    )
    target = min(args.num_random_trials, max_combinations)
    attempts = 0
    while len(trials) < target and attempts < max_combinations * 4:
        attempts += 1
        trial = make_trial(
            profile=rng.choice(args.tune_profiles),
            conv_hidden_dim=rng.choice(space["conv_hidden_dims"]),
            conv_grid_size=rng.choice(space["conv_grid_sizes"]),
            head_hidden_dim=rng.choice(space["head_hidden_dims"]),
            head_grid_size=rng.choice(space["head_grid_sizes"]),
            lr=rng.choice(space["lrs"]),
            weight_decay=rng.choice(space["weight_decays"]),
            dropout=rng.choice(space["dropouts"]),
        )
        trials.setdefault(trial_key(trial), trial)
    return list(trials.values())


def grid_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    import itertools

    space = candidate_space(args)
    trials = []
    for values in itertools.product(
        args.tune_profiles,
        space["conv_hidden_dims"],
        space["conv_grid_sizes"],
        space["head_hidden_dims"],
        space["head_grid_sizes"],
        space["lrs"],
        space["weight_decays"],
        space["dropouts"],
    ):
        trials.append(make_trial(*values))
    return trials


def make_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.search_space == "compact":
        trials = compact_trials(args)
    elif args.search_space == "grid":
        trials = grid_trials(args)
    else:
        trials = random_trials(args)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]
    return trials


def rung_schedule(args: argparse.Namespace) -> list[dict[str, Any]]:
    rung_epochs = args.rung_epochs or [
        max(1, args.epochs // 10),
        max(2, int(round(args.epochs * 0.3))),
        args.epochs,
    ]
    fold_counts = args.rung_fold_counts or [
        min(2, len(args.folds)),
        min(3, len(args.folds)),
        len(args.folds),
    ]
    train_sizes = args.rung_train_sizes or [512, 0, 0]
    if not (len(rung_epochs) == len(fold_counts) == len(train_sizes)):
        raise ValueError("--rung-epochs, --rung-fold-counts, and --rung-train-sizes must have equal lengths")
    if args.halving_factor < 2:
        raise ValueError("--halving-factor must be at least 2")
    schedule = []
    for idx, (epochs, fold_count, train_size) in enumerate(
        zip(rung_epochs, fold_counts, train_sizes),
        start=1,
    ):
        if fold_count < 1 or fold_count > len(args.folds):
            raise ValueError(f"rung {idx} fold count {fold_count} is outside 1..{len(args.folds)}")
        schedule.append(
            {
                "name": f"rung{idx}_e{epochs}_f{fold_count}",
                "epochs": epochs,
                "folds": args.folds[:fold_count],
                "train_size": None if train_size <= 0 else train_size,
            }
        )
    return schedule


def stage_args(
    args: argparse.Namespace,
    stage_name: str,
    epochs: int,
    train_size: int | None,
    val_ratio: float | None = None,
) -> argparse.Namespace:
    values = vars(args).copy()
    values["stage_name"] = stage_name
    values["epochs"] = epochs
    values["train_size"] = train_size
    if val_ratio is not None:
        values["val_ratio"] = val_ratio
    return argparse.Namespace(**values)


def head_dims_for_trial(args: argparse.Namespace, trial: dict[str, Any]) -> tuple[list[int], list[int]]:
    if trial["profile"] == "kan-readout":
        return [int(trial["head_kan_hidden_dim"])], [int(trial["head_kan_hidden_dim"])]
    if trial["profile"] == "kan-conv":
        return list(args.head_hidden_dims), list(args.head_hidden_dims)
    return list(args.head_hidden_dims), [int(trial["head_kan_hidden_dim"])]


def run_trial_fold(
    args: argparse.Namespace,
    output_dir: Path,
    trial: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    stage_name = getattr(args, "stage_name", "full")
    run_dir = output_dir / stage_name / trial["trial_id"] / f"fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    head_hidden_dims, kan_head_hidden_dims = head_dims_for_trial(args, trial)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_matbench.py"),
        "--dataset",
        args.dataset,
        "--fold",
        str(fold),
        "--conv-nets",
        trial["conv_net"],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--head-hidden-dims",
        *[str(dim) for dim in head_hidden_dims],
        "--kan-head-hidden-dims",
        *[str(dim) for dim in kan_head_hidden_dims],
        "--mlp-head-net",
        trial["mlp_head_net"],
        "--kan-head-net",
        trial["kan_head_net"],
        "--num-convs",
        str(args.num_convs),
        "--conv-kan-impl",
        args.conv_kan_impl,
        "--conv-kan-hidden-dim",
        str(trial["conv_kan_hidden_dim"]),
        "--conv-kan-grid-size",
        str(trial["conv_kan_grid_size"]),
        "--head-kan-impl",
        args.head_kan_impl or args.conv_kan_impl,
        "--head-kan-grid-size",
        str(trial["head_kan_grid_size"]),
        "--edge-dim",
        str(args.edge_dim),
        "--cutoff",
        str(args.cutoff),
        "--mlp-atom-features",
        "cgcnn",
        "--mlp-edge-features",
        "gaussian",
        "--kan-atom-features",
        "cgcnn",
        "--kan-edge-features",
        "gaussian",
        "--val-ratio",
        str(args.val_ratio),
        "--dropout",
        str(trial["dropout"]),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--log-every-steps",
        str(args.log_every_steps),
        "--epoch-pause-seconds",
        str(args.epoch_pause_seconds),
        "--forward-iters",
        str(args.forward_iters),
        "--warmup-iters",
        str(args.warmup_iters),
        "--output-dir",
        str(run_dir),
    ]
    if trial["profile"] == "kan-readout":
        cmd.extend(["--mlp-lr", str(trial["lr"]), "--mlp-weight-decay", str(trial["weight_decay"])])
    else:
        cmd.extend(["--kan-lr", str(trial["lr"]), "--kan-weight-decay", str(trial["weight_decay"])])
    append_common_flags(cmd, args)

    print(f"\n=== {stage_name}: {trial['trial_id']} fold {fold} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    result = latest_result(run_dir, args.dataset, fold)
    return row_from_result(args, trial, result, fold)


def run_baseline_fold(
    args: argparse.Namespace,
    output_dir: Path,
    fold: int,
) -> dict[str, Any]:
    stage_name = getattr(args, "stage_name", "full")
    trial = {
        "trial_id": "cgcnn_baseline",
        "profile": "cgcnn",
        "comparison": "baseline",
        "conv_net": "mlp",
        "mlp_head_net": "mlp",
        "kan_head_net": "kan",
        "conv_kan_hidden_dim": "",
        "conv_kan_grid_size": "",
        "head_kan_hidden_dim": "",
        "head_kan_grid_size": "",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": 0.0,
    }
    run_dir = output_dir / stage_name / "cgcnn_baseline" / f"fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_matbench.py"),
        "--dataset",
        args.dataset,
        "--fold",
        str(fold),
        "--conv-nets",
        "mlp",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--head-hidden-dims",
        *[str(dim) for dim in args.head_hidden_dims],
        "--kan-head-hidden-dims",
        *[str(dim) for dim in args.head_hidden_dims],
        "--mlp-head-net",
        "mlp",
        "--kan-head-net",
        "kan",
        "--num-convs",
        str(args.num_convs),
        "--edge-dim",
        str(args.edge_dim),
        "--cutoff",
        str(args.cutoff),
        "--mlp-atom-features",
        "cgcnn",
        "--mlp-edge-features",
        "gaussian",
        "--kan-atom-features",
        "cgcnn",
        "--kan-edge-features",
        "gaussian",
        "--val-ratio",
        str(args.val_ratio),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--log-every-steps",
        str(args.log_every_steps),
        "--epoch-pause-seconds",
        str(args.epoch_pause_seconds),
        "--forward-iters",
        str(args.forward_iters),
        "--warmup-iters",
        str(args.warmup_iters),
        "--output-dir",
        str(run_dir),
    ]
    append_common_flags(cmd, args)
    print(f"\n=== {stage_name}: cgcnn_baseline fold {fold} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    result = latest_result(run_dir, args.dataset, fold)
    return row_from_result(args, trial, result, fold)


def append_common_flags(cmd: list[str], args: argparse.Namespace) -> None:
    if args.train_size is not None:
        cmd.extend(["--train-size", str(args.train_size)])
    if args.test_size is not None:
        cmd.extend(["--test-size", str(args.test_size)])
    if args.require_cuda:
        cmd.append("--require-cuda")
    if args.pin_memory:
        cmd.append("--pin-memory")
    if args.persistent_workers:
        cmd.append("--persistent-workers")


def latest_result(run_dir: Path, dataset: str, fold: int) -> dict[str, Any]:
    paths = sorted(run_dir.glob(f"matbench-{dataset}-fold{fold}-*.json"))
    if not paths:
        raise FileNotFoundError(f"No result JSON found in {run_dir}")
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    return payload["results"][0]


def row_from_result(
    args: argparse.Namespace,
    trial: dict[str, Any],
    result: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    return {
        "trial_id": trial["trial_id"],
        "profile": trial["profile"],
        "comparison": trial["comparison"],
        "stage": getattr(args, "stage_name", "full"),
        "fold": fold,
        "epochs": args.epochs,
        "train_size": args.train_size if args.train_size is not None else "all",
        "conv_net": result["conv_net"],
        "head_net": result["head_net"],
        "head_hidden_dims": result["head_hidden_dims"],
        "head_kan_grid_size": result.get("head_kan_grid_size", ""),
        "atom_features": result.get("atom_features", ""),
        "edge_features": result.get("edge_features", ""),
        "params": int(result["params"]),
        "best_val_mae": metric_float(result["best_val_mae"]),
        "test_mae": metric_float(result["test_mae"]),
        "test_rmse": metric_float(result["test_rmse"]),
        "train_seconds": metric_float(result["train_seconds"]),
        "forward_ms_per_batch": metric_float(result["forward_ms_per_batch"]),
        "conv_kan_hidden_dim": trial["conv_kan_hidden_dim"],
        "conv_kan_grid_size": trial["conv_kan_grid_size"],
        "trial_head_kan_hidden_dim": trial["head_kan_hidden_dim"],
        "trial_head_kan_grid_size": trial["head_kan_grid_size"],
        "trial_lr": trial["lr"],
        "trial_weight_decay": trial["weight_decay"],
        "dropout": trial["dropout"],
    }


def metric_float(value: Any) -> float:
    number = float(value)
    if math.isnan(number):
        return float("inf")
    return number


def summary_stats(values: list[float]) -> tuple[float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("inf"), float("inf")
    if len(finite_values) == 1:
        return finite_values[0], 0.0
    avg = mean(finite_values)
    variance = sum((value - avg) ** 2 for value in finite_values) / (len(finite_values) - 1)
    return avg, math.sqrt(variance)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_trial: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_trial.setdefault(row["trial_id"], []).append(row)

    summary_rows = []
    for trial_id, trial_rows in by_trial.items():
        first = trial_rows[0]
        summary = {
            "trial_id": trial_id,
            "profile": first["profile"],
            "comparison": first["comparison"],
            "stage": first["stage"],
            "folds": len(trial_rows),
            "epochs": first["epochs"],
            "train_size": first["train_size"],
            "conv_net": first["conv_net"],
            "head_net": first["head_net"],
            "head_hidden_dims": first["head_hidden_dims"],
            "atom_features": first["atom_features"],
            "edge_features": first["edge_features"],
            "conv_kan_hidden_dim": first["conv_kan_hidden_dim"],
            "conv_kan_grid_size": first["conv_kan_grid_size"],
            "trial_head_kan_hidden_dim": first["trial_head_kan_hidden_dim"],
            "trial_head_kan_grid_size": first["trial_head_kan_grid_size"],
            "trial_lr": first["trial_lr"],
            "trial_weight_decay": first["trial_weight_decay"],
            "dropout": first["dropout"],
        }
        for metric in ["params", *METRICS]:
            values = [float(row[metric]) for row in trial_rows]
            metric_mean, metric_std = summary_stats(values)
            summary[f"{metric}_mean"] = metric_mean
            summary[f"{metric}_std"] = metric_std
        summary_rows.append(summary)
    return summary_rows


def run_full_strategy(
    args: argparse.Namespace,
    output_dir: Path,
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    full_args = stage_args(args, "full", args.epochs, args.train_size)
    for fold in args.folds:
        rows.append(run_baseline_fold(full_args, output_dir, fold))
    for trial in trials:
        for fold in args.folds:
            rows.append(run_trial_fold(full_args, output_dir, trial, fold))
    return rows, summarize_rows(rows)


def run_successive_halving(
    args: argparse.Namespace,
    output_dir: Path,
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schedule = rung_schedule(args)
    metric_mean = f"{args.metric}_mean"
    all_rows = []
    active_trials = list(trials)
    trials_by_id = {trial["trial_id"]: trial for trial in trials}
    for idx, stage in enumerate(schedule, start=1):
        print(
            f"\n### Rung {idx}/{len(schedule)}: epochs={stage['epochs']}, "
            f"folds={stage['folds']}, train_size={stage['train_size'] or 'all'}, "
            f"active_trials={len(active_trials)}",
            flush=True,
        )
        current_args = stage_args(args, stage["name"], stage["epochs"], stage["train_size"])
        stage_rows = []
        for trial in active_trials:
            for fold in stage["folds"]:
                stage_rows.append(run_trial_fold(current_args, output_dir, trial, fold))
        if idx == len(schedule):
            for fold in stage["folds"]:
                stage_rows.append(run_baseline_fold(current_args, output_dir, fold))
        all_rows.extend(stage_rows)
        stage_summary = summarize_rows(stage_rows)
        stage_summary.sort(key=lambda row: float(row[metric_mean]))
        write_csv(output_dir / f"{stage['name']}-summary-{args.dataset}.csv", stage_summary)
        print_summary(stage_summary, args.metric)
        if idx < len(schedule):
            ranked_rows = [
                row
                for row in stage_summary
                if row["profile"] != "cgcnn"
            ]
            survivors = max(1, math.ceil(len(active_trials) / args.halving_factor))
            survivor_ids = [row["trial_id"] for row in ranked_rows[:survivors]]
            for profile in args.tune_profiles:
                best_profile_row = next(
                    (row for row in ranked_rows if row["profile"] == profile),
                    None,
                )
                if best_profile_row is not None:
                    survivor_ids.append(best_profile_row["trial_id"])
            survivor_ids = list(dict.fromkeys(survivor_ids))
            active_trials = [trials_by_id[trial_id] for trial_id in survivor_ids]
            print(
                f"Keeping {len(active_trials)} / {len(ranked_rows)} trials for next rung: "
                f"{[trial['trial_id'] for trial in active_trials]}",
                flush=True,
            )
    final_stage = schedule[-1]["name"]
    final_rows = [row for row in all_rows if row["stage"] == final_stage]
    return all_rows, summarize_rows(final_rows)


def best_by_profile(summary_rows: list[dict[str, Any]], metric: str) -> dict[str, dict[str, Any]]:
    metric_mean = f"{metric}_mean"
    best: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        if row["profile"] == "cgcnn":
            continue
        current = best.get(row["profile"])
        if current is None or float(row[metric_mean]) < float(current[metric_mean]):
            best[row["profile"]] = row
    return best


def trial_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    profile = str(summary["profile"])
    settings = profile_settings(profile)
    return {
        "trial_id": f"final_{summary['trial_id']}",
        "profile": profile,
        **settings,
        "conv_kan_hidden_dim": int(float(summary["conv_kan_hidden_dim"])),
        "conv_kan_grid_size": int(float(summary["conv_kan_grid_size"])),
        "head_kan_hidden_dim": int(float(summary["trial_head_kan_hidden_dim"])),
        "head_kan_grid_size": int(float(summary["trial_head_kan_grid_size"])),
        "lr": float(summary["trial_lr"]),
        "weight_decay": float(summary["trial_weight_decay"]),
        "dropout": float(summary["dropout"]),
    }


def run_final_benchmark(
    args: argparse.Namespace,
    output_dir: Path,
    selected: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_dir = output_dir / "final-benchmark"
    final_args = stage_args(
        args,
        "final_benchmark",
        args.epochs,
        args.train_size,
        val_ratio=args.final_val_ratio,
    )
    rows = []
    for fold in args.folds:
        rows.append(run_baseline_fold(final_args, final_dir, fold))
    for profile in KAN_PROFILES:
        if profile not in selected:
            continue
        trial = trial_from_summary(selected[profile])
        for fold in args.folds:
            rows.append(run_trial_fold(final_args, final_dir, trial, fold))
    summary = summarize_rows(rows)
    summary.sort(key=lambda row: float(row["test_mae_mean"]))
    write_csv(output_dir / f"final-benchmark-fold-results-{args.dataset}.csv", rows)
    write_csv(output_dir / f"final-benchmark-summary-{args.dataset}.csv", summary)
    (output_dir / "final-benchmark-summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "folds": args.folds,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: list[dict[str, Any]], metric: str) -> None:
    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"
    headers = [
        "rank",
        "profile",
        "trial_id",
        "folds",
        "params_mean",
        metric_mean,
        metric_std,
    ]
    if metric != "test_mae":
        headers.append("test_mae_mean")
    headers.append("forward_ms_per_batch_mean")
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


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_float_id(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    if args.val_ratio <= 0 and args.metric == "best_val_mae":
        raise ValueError("--metric best_val_mae requires --val-ratio > 0")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "benchmarks" / f"tune-fair-{args.dataset}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    trials = make_trials(args)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Tune profiles: {args.tune_profiles}", flush=True)
    print(f"Trials: {len(trials)}", flush=True)
    print(f"Strategy: {args.strategy}", flush=True)
    print(f"Metric: {args.metric}", flush=True)
    if args.dry_run:
        payload = {
            "trials": trials,
            "rungs": rung_schedule(args) if args.strategy == "successive-halving" else None,
            "run_final_benchmark": args.run_final_benchmark,
        }
        print(json.dumps(payload, indent=2), flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.strategy == "successive-halving":
        rows, summary_rows = run_successive_halving(args, output_dir, trials)
    else:
        rows, summary_rows = run_full_strategy(args, output_dir, trials)

    metric_mean = f"{args.metric}_mean"
    summary_rows.sort(key=lambda row: float(row[metric_mean]))
    selected = best_by_profile(summary_rows, args.metric)
    best_overall = summary_rows[0]
    baseline = next((row for row in summary_rows if row["profile"] == "cgcnn"), None)

    write_csv(output_dir / f"tuning-fold-results-{args.dataset}.csv", rows)
    write_csv(output_dir / f"tuning-summary-{args.dataset}.csv", summary_rows)
    best_payload = {
        "dataset": args.dataset,
        "folds": args.folds,
        "selection_metric": args.metric,
        "best_by_profile": selected,
        "best_overall": best_overall,
        "cgcnn_baseline": baseline,
        "all_trials": summary_rows,
    }
    (output_dir / "best_config.json").write_text(
        json.dumps(best_payload, indent=2),
        encoding="utf-8",
    )

    print()
    print_summary(summary_rows, args.metric)
    print(f"\nBest overall: {best_overall['profile']} / {best_overall['trial_id']}")
    for profile, row in selected.items():
        print(f"Best {profile}: {row['trial_id']} ({metric_mean}={row[metric_mean]:.6g})")
    if baseline is not None:
        print(f"CGCNN baseline: {metric_mean}={baseline[metric_mean]:.6g}")

    if args.run_final_benchmark:
        _, final_summary = run_final_benchmark(args, output_dir, selected)
        print("\nFinal benchmark summary:")
        print_summary(final_summary, "test_mae")

    print(f"\nWrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
