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
ATOM_FEATURE_CHOICES = ["onehot", "atomic_number", "elemental", "cgcnn"]
EDGE_FEATURE_CHOICES = ["gaussian", "distance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune FastKAN CGCNN hyperparameters across Matbench folds."
    )
    parser.add_argument("--dataset", default="matbench_phonons")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dims", type=int, nargs="+", default=[32])
    parser.add_argument("--kan-head-hidden-dims", type=int, nargs="+", default=[8])
    parser.add_argument("--kan-head-hidden-dim-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--mlp-head-net", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--kan-head-net", choices=["mlp", "kan"], default="kan")
    parser.add_argument("--num-convs", type=int, default=4)
    parser.add_argument("--edge-dim", type=int, default=41)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument(
        "--atom-features",
        choices=ATOM_FEATURE_CHOICES,
        default=None,
        help="Backward-compatible alias for --kan-atom-features.",
    )
    parser.add_argument(
        "--edge-features",
        choices=EDGE_FEATURE_CHOICES,
        default=None,
        help="Backward-compatible alias for --kan-edge-features.",
    )
    parser.add_argument(
        "--mlp-atom-features",
        choices=ATOM_FEATURE_CHOICES,
        default="cgcnn",
    )
    parser.add_argument("--mlp-edge-features", choices=EDGE_FEATURE_CHOICES, default="gaussian")
    parser.add_argument(
        "--kan-atom-features",
        choices=ATOM_FEATURE_CHOICES,
        default="cgcnn",
    )
    parser.add_argument("--kan-edge-features", choices=EDGE_FEATURE_CHOICES, default="gaussian")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--conv-kan-impl", choices=["fastkan", "spline"], default="fastkan")
    parser.add_argument("--conv-kan-hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--conv-kan-grid-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--kan-lrs", type=float, nargs="+", default=None)
    parser.add_argument("--kan-weight-decays", type=float, nargs="+", default=None)
    parser.add_argument("--dropouts", type=float, nargs="+", default=None)
    parser.add_argument("--search-space", choices=["random", "compact", "grid"], default="random")
    parser.add_argument("--num-random-trials", type=int, default=12)
    parser.add_argument("--strategy", choices=["successive-halving", "full"], default="successive-halving")
    parser.add_argument("--halving-factor", type=int, default=3)
    parser.add_argument("--rung-epochs", type=int, nargs="+", default=None)
    parser.add_argument("--rung-fold-counts", type=int, nargs="+", default=None)
    parser.add_argument("--rung-train-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--metric", choices=["best_val_mae", "test_mae", "test_rmse"], default="best_val_mae")
    parser.add_argument("--include-mlp-baseline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-mlp-baseline",
        action="store_true",
        help="Skip benchmarking the ordinary CGCNN baseline in the final tuning rung.",
    )
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.search_space == "compact":
        return compact_trials(args)
    if args.search_space == "grid":
        return grid_trials(args)
    return random_trials(args)


def apply_feature_aliases(args: argparse.Namespace) -> None:
    if args.atom_features is not None:
        args.kan_atom_features = args.atom_features
    if args.edge_features is not None:
        args.kan_edge_features = args.edge_features


def benchmark_mlp_baseline(args: argparse.Namespace) -> bool:
    return not args.skip_mlp_baseline


def compact_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    head_dims = args.kan_head_hidden_dim_candidates or [8, 16, 32]

    def head_dim(index: int) -> int:
        return head_dims[min(index, len(head_dims) - 1)]

    return [
        make_trial(8, head_dim(0), 3, 3e-3, 0.0, 0.0),
        make_trial(16, head_dim(0), 2, 3e-3, 0.0, 0.0),
        make_trial(16, head_dim(1), 3, 1e-3, 0.0, 0.0),
        make_trial(16, head_dim(1), 3, 3e-3, 0.0, 0.0),
        make_trial(24, head_dim(1), 2, 1e-3, 0.0, 0.0),
        make_trial(16, head_dim(2), 4, 1e-3, 0.0, 0.0),
    ]


def random_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    hidden_dims = args.conv_kan_hidden_dims or [4, 8, 12, 16, 24, 32]
    head_hidden_dims = args.kan_head_hidden_dim_candidates or [8, 16, 32]
    grid_sizes = args.conv_kan_grid_sizes or [2, 3, 4]
    lrs = args.kan_lrs or [5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
    weight_decays = args.kan_weight_decays or [0.0, 1e-6, 1e-5]
    dropouts = args.dropouts or [0.0, 0.05, 0.1]

    rng = random.Random(args.seed)
    trials: dict[tuple[int, int, float, float, float], dict[str, Any]] = {}

    # Seed the random search with a few sensible parameter-matched candidates.
    for trial in compact_trials(args):
        key = trial_key(trial)
        trials[key] = trial
        if len(trials) >= args.num_random_trials:
            return list(trials.values())

    max_combinations = (
        len(hidden_dims)
        * len(head_hidden_dims)
        * len(grid_sizes)
        * len(lrs)
        * len(weight_decays)
        * len(dropouts)
    )
    target = min(args.num_random_trials, max_combinations)
    attempts = 0
    while len(trials) < target and attempts < max_combinations * 4:
        attempts += 1
        trial = make_trial(
            hidden_dim=rng.choice(hidden_dims),
            head_hidden_dim=rng.choice(head_hidden_dims),
            grid_size=rng.choice(grid_sizes),
            lr=rng.choice(lrs),
            weight_decay=rng.choice(weight_decays),
            dropout=rng.choice(dropouts),
        )
        trials.setdefault(trial_key(trial), trial)
    return list(trials.values())


def grid_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    hidden_dims = args.conv_kan_hidden_dims or [8, 16, 24]
    head_hidden_dims = args.kan_head_hidden_dim_candidates or [8, 16, 32]
    grid_sizes = args.conv_kan_grid_sizes or [2, 3]
    lrs = args.kan_lrs or [1e-3, 3e-3]
    weight_decays = args.kan_weight_decays or [0.0]
    dropouts = args.dropouts or [0.0]

    trials = []
    for hidden_dim, head_hidden_dim, grid_size, lr, weight_decay, dropout in itertools.product(
        hidden_dims,
        head_hidden_dims,
        grid_sizes,
        lrs,
        weight_decays,
        dropouts,
    ):
        trials.append(make_trial(hidden_dim, head_hidden_dim, grid_size, lr, weight_decay, dropout))
    return trials


def make_trial(
    hidden_dim: int,
    head_hidden_dim: int,
    grid_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
) -> dict[str, Any]:
    trial_id = (
        f"kan_h{hidden_dim}_rh{head_hidden_dim}_g{grid_size}_"
        f"lr{format_float_id(lr)}_wd{format_float_id(weight_decay)}_"
        f"do{format_float_id(dropout)}"
    )
    return {
        "trial_id": trial_id,
        "conv_kan_hidden_dim": hidden_dim,
        "kan_head_hidden_dims": [head_hidden_dim],
        "conv_kan_grid_size": grid_size,
        "kan_lr": lr,
        "kan_weight_decay": weight_decay,
        "dropout": dropout,
    }


def trial_key(trial: dict[str, Any]) -> tuple[int, tuple[int, ...], int, float, float, float]:
    return (
        int(trial["conv_kan_hidden_dim"]),
        tuple(int(dim) for dim in trial["kan_head_hidden_dims"]),
        int(trial["conv_kan_grid_size"]),
        float(trial["kan_lr"]),
        float(trial["kan_weight_decay"]),
        float(trial["dropout"]),
    )


def format_float_id(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def safe_float(value: Any) -> float:
    number = float(value)
    if math.isnan(number):
        return float("inf")
    return number


def stage_args(
    args: argparse.Namespace,
    stage_name: str,
    epochs: int,
    train_size: int | None,
) -> argparse.Namespace:
    values = vars(args).copy()
    values["stage_name"] = stage_name
    values["epochs"] = epochs
    if train_size is not None:
        values["train_size"] = train_size
    return argparse.Namespace(**values)


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
        stage_train_size = None if train_size <= 0 else train_size
        stage_folds = args.folds[:fold_count]
        schedule.append(
            {
                "name": f"rung{idx}_e{epochs}_f{fold_count}",
                "epochs": epochs,
                "folds": stage_folds,
                "train_size": stage_train_size,
            }
        )
    return schedule


def run_trial_fold(
    args: argparse.Namespace,
    output_dir: Path,
    trial: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    stage_name = getattr(args, "stage_name", "full")
    run_dir = output_dir / stage_name / trial["trial_id"] / f"fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_matbench.py"),
        "--dataset",
        args.dataset,
        "--fold",
        str(fold),
        "--conv-nets",
        "kan",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--head-hidden-dims",
        *[str(dim) for dim in args.head_hidden_dims],
        "--kan-head-hidden-dims",
        *[str(dim) for dim in trial["kan_head_hidden_dims"]],
        "--mlp-head-net",
        args.mlp_head_net,
        "--kan-head-net",
        args.kan_head_net,
        "--num-convs",
        str(args.num_convs),
        "--conv-kan-impl",
        args.conv_kan_impl,
        "--conv-kan-hidden-dim",
        str(trial["conv_kan_hidden_dim"]),
        "--conv-kan-grid-size",
        str(trial["conv_kan_grid_size"]),
        "--edge-dim",
        str(args.edge_dim),
        "--cutoff",
        str(args.cutoff),
        "--mlp-atom-features",
        args.mlp_atom_features,
        "--mlp-edge-features",
        args.mlp_edge_features,
        "--kan-atom-features",
        args.kan_atom_features,
        "--kan-edge-features",
        args.kan_edge_features,
        "--val-ratio",
        str(args.val_ratio),
        "--dropout",
        str(trial["dropout"]),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--kan-lr",
        str(trial["kan_lr"]),
        "--kan-weight-decay",
        str(trial["kan_weight_decay"]),
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

    print(f"\n=== {trial['trial_id']} fold {fold} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    paths = sorted(run_dir.glob(f"matbench-{args.dataset}-fold{fold}-*.json"))
    if not paths:
        raise FileNotFoundError(f"No result JSON found in {run_dir}")
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    result = payload["results"][0]
    return {
        "trial_id": trial["trial_id"],
        "stage": stage_name,
        "fold": fold,
        "epochs": args.epochs,
        "train_size": args.train_size if args.train_size is not None else "all",
        "params": int(result["params"]),
        "best_val_mae": safe_float(result["best_val_mae"]),
        "test_mae": safe_float(result["test_mae"]),
        "test_rmse": safe_float(result["test_rmse"]),
        "train_seconds": safe_float(result["train_seconds"]),
        "forward_ms_per_batch": safe_float(result["forward_ms_per_batch"]),
        **trial,
    }


def run_mlp_baseline_fold(
    args: argparse.Namespace,
    output_dir: Path,
    fold: int,
) -> dict[str, Any]:
    trial_id = "mlp_baseline"
    stage_name = getattr(args, "stage_name", "full")
    run_dir = output_dir / stage_name / trial_id / f"fold{fold}"
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
        *[str(dim) for dim in args.kan_head_hidden_dims],
        "--mlp-head-net",
        args.mlp_head_net,
        "--kan-head-net",
        args.kan_head_net,
        "--num-convs",
        str(args.num_convs),
        "--edge-dim",
        str(args.edge_dim),
        "--cutoff",
        str(args.cutoff),
        "--mlp-atom-features",
        args.mlp_atom_features,
        "--mlp-edge-features",
        args.mlp_edge_features,
        "--kan-atom-features",
        args.kan_atom_features,
        "--kan-edge-features",
        args.kan_edge_features,
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

    print(f"\n=== {trial_id} fold {fold} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    paths = sorted(run_dir.glob(f"matbench-{args.dataset}-fold{fold}-*.json"))
    if not paths:
        raise FileNotFoundError(f"No result JSON found in {run_dir}")
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    result = payload["results"][0]
    return {
        "trial_id": trial_id,
        "stage": stage_name,
        "fold": fold,
        "epochs": args.epochs,
        "train_size": args.train_size if args.train_size is not None else "all",
        "params": int(result["params"]),
        "best_val_mae": safe_float(result["best_val_mae"]),
        "test_mae": safe_float(result["test_mae"]),
        "test_rmse": safe_float(result["test_rmse"]),
        "train_seconds": safe_float(result["train_seconds"]),
        "forward_ms_per_batch": safe_float(result["forward_ms_per_batch"]),
        "conv_kan_hidden_dim": "",
        "kan_head_hidden_dims": "",
        "conv_kan_grid_size": "",
        "kan_lr": "",
        "kan_weight_decay": "",
        "dropout": "",
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_trial: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_trial.setdefault(row["trial_id"], []).append(row)

    summary_rows = []
    for trial_id, trial_rows in by_trial.items():
        first = trial_rows[0]
        summary = {
            "trial_id": trial_id,
            "stage": first.get("stage", "full"),
            "folds": len(trial_rows),
            "epochs": first.get("epochs", ""),
            "train_size": first.get("train_size", ""),
            "conv_kan_hidden_dim": first["conv_kan_hidden_dim"],
            "kan_head_hidden_dims": first["kan_head_hidden_dims"],
            "conv_kan_grid_size": first["conv_kan_grid_size"],
            "kan_lr": first["kan_lr"],
            "kan_weight_decay": first["kan_weight_decay"],
            "dropout": first["dropout"],
            "params_mean": mean(float(row["params"]) for row in trial_rows),
        }
        for metric in (
            "best_val_mae",
            "test_mae",
            "test_rmse",
            "train_seconds",
            "forward_ms_per_batch",
        ):
            values = [float(row[metric]) for row in trial_rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def run_full_strategy(
    args: argparse.Namespace,
    output_dir: Path,
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    full_args = stage_args(args, "full", args.epochs, args.train_size)
    if benchmark_mlp_baseline(args):
        for fold in args.folds:
            rows.append(run_mlp_baseline_fold(full_args, output_dir, fold))

    for trial in trials:
        for fold in args.folds:
            rows.append(run_trial_fold(full_args, output_dir, trial, fold))

    summary_rows = summarize_rows(rows)
    return rows, summary_rows


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

        if benchmark_mlp_baseline(args) and idx == len(schedule):
            for fold in stage["folds"]:
                stage_rows.append(run_mlp_baseline_fold(current_args, output_dir, fold))

        all_rows.extend(stage_rows)
        stage_summary = summarize_rows(stage_rows)
        stage_summary.sort(key=lambda row: float(row[metric_mean]))
        write_csv(output_dir / f"{stage['name']}-summary-{args.dataset}.csv", stage_summary)
        print_summary(stage_summary, args.metric)

        if idx < len(schedule):
            ranked_trial_ids = [
                row["trial_id"]
                for row in stage_summary
                if row["trial_id"] != "mlp_baseline"
            ]
            survivors = max(1, math.ceil(len(active_trials) / args.halving_factor))
            active_trials = [trials_by_id[trial_id] for trial_id in ranked_trial_ids[:survivors]]
            print(
                f"Keeping {len(active_trials)} / {len(ranked_trial_ids)} trials for next rung: "
                f"{[trial['trial_id'] for trial in active_trials]}",
                flush=True,
            )

    final_stage = schedule[-1]["name"]
    final_rows = [row for row in all_rows if row["stage"] == final_stage]
    final_summary = summarize_rows(final_rows)
    return all_rows, final_summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def final_command(args: argparse.Namespace, best: dict[str, Any]) -> list[str]:
    cmd = [
        "python",
        "scripts/benchmark_matbench_5fold.py",
        "--dataset",
        args.dataset,
        "--folds",
        *[str(fold) for fold in args.folds],
        "--conv-nets",
        "mlp",
        "kan",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--head-hidden-dims",
        *[str(dim) for dim in args.head_hidden_dims],
        "--kan-head-hidden-dims",
        *[str(dim) for dim in best["kan_head_hidden_dims"]],
        "--mlp-head-net",
        args.mlp_head_net,
        "--kan-head-net",
        args.kan_head_net,
        "--num-convs",
        str(args.num_convs),
        "--conv-kan-hidden-dim",
        str(best["conv_kan_hidden_dim"]),
        "--conv-kan-grid-size",
        str(best["conv_kan_grid_size"]),
        "--conv-kan-impl",
        args.conv_kan_impl,
        "--edge-dim",
        str(args.edge_dim),
        "--cutoff",
        str(args.cutoff),
        "--mlp-atom-features",
        args.mlp_atom_features,
        "--mlp-edge-features",
        args.mlp_edge_features,
        "--kan-atom-features",
        args.kan_atom_features,
        "--kan-edge-features",
        args.kan_edge_features,
        "--kan-lr",
        str(best["kan_lr"]),
        "--kan-weight-decay",
        str(best["kan_weight_decay"]),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if args.require_cuda:
        cmd.append("--require-cuda")
    return cmd


def print_summary(summary_rows: list[dict[str, Any]], metric: str) -> None:
    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"
    headers = [
        "rank",
        "trial_id",
        "folds",
        "params_mean",
        "kan_head_hidden_dims",
        metric_mean,
        metric_std,
        "test_mae_mean",
        "forward_ms_per_batch_mean",
    ]
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
        values = {
            "rank": idx,
            **row,
        }
        print(" | ".join(format_value(values.get(header, "")).ljust(widths[header]) for header in headers))


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    args = parse_args()
    apply_feature_aliases(args)
    if args.val_ratio <= 0 and args.metric == "best_val_mae":
        raise ValueError("--metric best_val_mae requires --val-ratio > 0")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT
        / "benchmarks"
        / f"tune-{args.dataset}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = make_trials(args)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    print(f"Output dir: {output_dir}", flush=True)
    print(f"Folds: {args.folds}", flush=True)
    print(f"Metric: {args.metric}", flush=True)
    print(f"Strategy: {args.strategy}", flush=True)
    print(f"Search space: {args.search_space}", flush=True)
    print(f"KAN trials: {len(trials)}", flush=True)
    print(f"Benchmark MLP baseline in final rung: {benchmark_mlp_baseline(args)}", flush=True)

    if args.dry_run:
        dry_run_payload = {
            "strategy": args.strategy,
            "search_space": args.search_space,
            "benchmark_mlp_baseline": benchmark_mlp_baseline(args),
            "trials": trials,
            "rungs": rung_schedule(args) if args.strategy == "successive-halving" else None,
        }
        print(json.dumps(dry_run_payload, indent=2), flush=True)
        return

    if args.strategy == "successive-halving":
        rows, summary_rows = run_successive_halving(args, output_dir, trials)
    else:
        rows, summary_rows = run_full_strategy(args, output_dir, trials)

    fold_csv = output_dir / f"fold-results-{args.dataset}.csv"
    write_csv(fold_csv, rows)

    metric_mean = f"{args.metric}_mean"
    summary_rows.sort(key=lambda row: float(row[metric_mean]))
    summary_csv = output_dir / f"tuning-summary-{args.dataset}.csv"
    write_csv(summary_csv, summary_rows)

    kan_summary_rows = [row for row in summary_rows if row["trial_id"] != "mlp_baseline"]
    if not kan_summary_rows:
        raise RuntimeError("No KAN trial summaries were produced")
    best_kan = min(kan_summary_rows, key=lambda row: float(row[metric_mean]))
    mlp_baseline = next(
        (row for row in summary_rows if row["trial_id"] == "mlp_baseline"),
        None,
    )
    best_overall = summary_rows[0]
    best_payload = {
        "dataset": args.dataset,
        "folds": args.folds,
        "selection_metric": args.metric,
        "best_kan": best_kan,
        "mlp_baseline": mlp_baseline,
        "best_overall_in_tuning_summary": best_overall,
        "final_5fold_command": final_command(args, best_kan),
        "all_trials": summary_rows,
    }
    best_json = output_dir / "best_config.json"
    best_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    print()
    print_summary(summary_rows, args.metric)
    print(f"\nBest KAN trial: {best_kan['trial_id']} ({metric_mean}={best_kan[metric_mean]:.6g})")
    if mlp_baseline is not None:
        print(f"MLP baseline: {metric_mean}={mlp_baseline[metric_mean]:.6g}")
        if float(mlp_baseline[metric_mean]) < float(best_kan[metric_mean]):
            print("Baseline is better than the best KAN trial on the selection metric.")
        else:
            print("Best KAN trial is better than or tied with the baseline on the selection metric.")
    print(f"Wrote {fold_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {best_json}")
    print("\nFinal 5-fold comparison command:")
    print(" ".join(best_payload["final_5fold_command"]))


if __name__ == "__main__":
    main()
