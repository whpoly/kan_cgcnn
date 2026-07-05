from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_head_helpers():
    helper_path = ROOT / "scripts" / "benchmark_jarvis_mace_heads.py"
    spec = importlib.util.spec_from_file_location("benchmark_jarvis_mace_heads", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_head_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-search tuning for MLP, KAN, and RandomForest heads on cached "
            "JARVIS MACE descriptors."
        )
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument(
        "--heads",
        nargs="+",
        default=["mlp", "fastkan", "spline", "rf"],
        choices=["mlp", "fastkan", "spline", "rf"],
    )
    parser.add_argument("--split", default="formula", choices=["random", "formula"])
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-trials-per-head", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--epoch-candidates",
        type=int,
        nargs="+",
        default=None,
        help="Candidate epoch budgets for MLP/KAN trials. Defaults to --epochs when omitted.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--tune-train-size", type=int, default=0)
    parser.add_argument("--tune-val-size", type=int, default=0)
    parser.add_argument("--no-refit-best", action="store_true")
    parser.add_argument("--log-every-epochs", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmarks/jarvis-mace-head-tuning")
    return parser.parse_args()


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def choose(rng: np.random.Generator, values: list[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def sample_indices(
    indices: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if size <= 0 or size >= len(indices):
        return indices
    selected = rng.choice(len(indices), size=size, replace=False)
    return indices[np.sort(selected)]


def sample_config(head: str, rng: np.random.Generator, args: argparse.Namespace) -> dict[str, Any]:
    batch_size = int(choose(rng, args.batch_sizes))
    epoch_candidates = args.epoch_candidates or [args.epochs]
    epochs = int(choose(rng, epoch_candidates))
    if head == "mlp":
        return {
            "label": "mlp",
            "fit_head": "mlp",
            "mlp_hidden_dims": choose(
                rng,
                [[64], [128], [256], [128, 64], [256, 128], [256, 128, 64]],
            ),
            "lr": log_uniform(rng, 3e-4, 3e-3),
            "weight_decay": choose(rng, [0.0, 1e-6, 1e-5, 1e-4, 1e-3]),
            "dropout": choose(rng, [0.0, 0.05, 0.1, 0.2]),
            "batch_size": batch_size,
            "epochs": epochs,
        }
    if head == "fastkan":
        return {
            "label": "fastkan",
            "fit_head": "kan",
            "kan_impl": "fastkan",
            "kan_hidden_dim": int(choose(rng, [16, 24, 32, 48, 64])),
            "kan_grid_size": int(choose(rng, [3, 4, 5])),
            "kan_spline_order": 3,
            "lr": log_uniform(rng, 2e-4, 3e-3),
            "weight_decay": choose(rng, [0.0, 1e-6, 1e-5, 1e-4, 1e-3]),
            "dropout": choose(rng, [0.0, 0.05, 0.1, 0.2]),
            "batch_size": batch_size,
            "epochs": epochs,
        }
    if head == "spline":
        return {
            "label": "spline",
            "fit_head": "kan",
            "kan_impl": "spline",
            "kan_hidden_dim": int(choose(rng, [12, 16, 18, 20, 24, 32])),
            "kan_grid_size": int(choose(rng, [3, 4])),
            "kan_spline_order": int(choose(rng, [2, 3])),
            "lr": log_uniform(rng, 1e-4, 2e-3),
            "weight_decay": choose(rng, [0.0, 1e-6, 1e-5, 1e-4]),
            "dropout": choose(rng, [0.0, 0.05, 0.1]),
            "batch_size": batch_size,
            "epochs": epochs,
        }
    if head == "rf":
        return {
            "label": "rf",
            "fit_head": "rf",
            "rf_n_estimators": int(choose(rng, [200, 400, 800])),
            "rf_max_depth": choose(rng, [None, 16, 32, 64]),
            "rf_min_samples_split": int(choose(rng, [2, 5, 10])),
            "rf_min_samples_leaf": int(choose(rng, [1, 2, 4])),
            "rf_max_features": choose(rng, ["sqrt", "log2", "0.33", "0.5", "1.0"]),
            "batch_size": batch_size,
        }
    raise ValueError(f"unsupported head {head!r}")


def build_trial_args(
    config: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> argparse.Namespace:
    return SimpleNamespace(
        seed=seed,
        device=args.device,
        num_workers=args.num_workers,
        log_every_epochs=args.log_every_epochs,
        epochs=int(config.get("epochs", args.epochs)),
        patience=args.patience,
        batch_size=int(config.get("batch_size", args.batch_sizes[0])),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        dropout=float(config.get("dropout", 0.0)),
        mlp_hidden_dims=list(config.get("mlp_hidden_dims", [128, 64])),
        kan_impl=str(config.get("kan_impl", "fastkan")),
        kan_hidden_dim=int(config.get("kan_hidden_dim", 32)),
        kan_grid_size=int(config.get("kan_grid_size", 3)),
        kan_spline_order=int(config.get("kan_spline_order", 3)),
        rf_n_estimators=int(config.get("rf_n_estimators", 300)),
        rf_max_depth=config.get("rf_max_depth"),
        rf_min_samples_split=int(config.get("rf_min_samples_split", 2)),
        rf_min_samples_leaf=int(config.get("rf_min_samples_leaf", 1)),
        rf_max_features=str(config.get("rf_max_features", "sqrt")),
        rf_n_jobs=args.rf_n_jobs,
        ridge_alpha=1.0,
    )


def run_config(
    config: dict[str, Any],
    trial_args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler,
) -> dict[str, Any]:
    helpers.set_seed(trial_args.seed)
    fit_head = str(config["fit_head"])
    if fit_head == "rf":
        row = helpers.fit_random_forest(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            target_scaler,
            trial_args,
        )
    else:
        row = helpers.fit_torch_head(
            fit_head,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            target_scaler,
            trial_args,
        )
    row["head"] = str(config["label"])
    return row


def compact_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"label", "fit_head"}}


def print_trial(row: dict[str, Any]) -> None:
    print(
        f"[trial {row['trial']:03d}] {row['head']} "
        f"val_mae={row['best_val_mae']:.6g} test_mae={row['test_mae']:.6g} "
        f"params={row['params']} config={row['config_json']}",
        flush=True,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "selected_trial",
        "head",
        "params",
        "best_val_mae",
        "train_mae",
        "test_mae",
        "test_rmse",
        "test_r2",
        "train_seconds",
        "epochs_ran",
        "config_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def main() -> None:
    args = parse_args()
    if args.val_ratio <= 0:
        raise ValueError("tuning needs a validation split; use --val-ratio > 0")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    loaded = helpers.load_embeddings(Path(args.embeddings))
    x = loaded["X"]
    y = loaded["y"]
    formulas = loaded["formulas"]
    split = helpers.split_indices(len(y), formulas, args)

    feature_scaler = helpers.Standardizer(x[split["train"]])
    target_scaler = helpers.TargetScaler(y[split["train"]])
    x_scaled = feature_scaler.transform(x)

    rng = np.random.default_rng(args.seed)
    tune_train_idx = sample_indices(split["train"], args.tune_train_size, rng)
    tune_val_idx = sample_indices(split["val"], args.tune_val_size, rng)

    x_train, y_train = x_scaled[tune_train_idx], y[tune_train_idx]
    x_val, y_val = x_scaled[tune_val_idx], y[tune_val_idx]
    x_test, y_test = x_scaled[split["test"]], y[split["test"]]
    full_train = (x_scaled[split["train"]], y[split["train"]])
    full_val = (x_scaled[split["val"]], y[split["val"]])

    print(
        f"Loaded {len(y)} embeddings, feature_dim={x.shape[1]}, split={args.split}, "
        f"full train/val/test={len(split['train'])}/{len(split['val'])}/{len(split['test'])}, "
        f"tune train/val={len(tune_train_idx)}/{len(tune_val_idx)}, device={device}",
        flush=True,
    )

    trial_rows: list[dict[str, Any]] = []
    trial_number = 0
    for head in args.heads:
        for _ in range(args.num_trials_per_head):
            trial_number += 1
            config = sample_config(head, rng, args)
            trial_args = build_trial_args(config, args, seed=args.seed + trial_number)
            start = time.perf_counter()
            row = run_config(
                config,
                trial_args,
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                target_scaler,
            )
            row["trial"] = trial_number
            row["selected_trial"] = ""
            row["config_json"] = json.dumps(compact_config(config), sort_keys=True)
            row["wall_seconds"] = time.perf_counter() - start
            trial_rows.append(row)
            print_trial(row)

    best_by_head: list[dict[str, Any]] = []
    for head in args.heads:
        candidates = [row for row in trial_rows if row["head"] == head]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row["best_val_mae"])
        best_by_head.append(best)

    final_rows: list[dict[str, Any]] = []
    if not args.no_refit_best:
        config_by_trial = {
            int(row["trial"]): json.loads(str(row["config_json"])) for row in trial_rows
        }
        label_by_trial = {int(row["trial"]): str(row["head"]) for row in trial_rows}
        fit_head_by_label = {"mlp": "mlp", "fastkan": "kan", "spline": "kan", "rf": "rf"}
        for best in best_by_head:
            selected_trial = int(best["trial"])
            label = label_by_trial[selected_trial]
            config = {
                "label": label,
                "fit_head": fit_head_by_label[label],
                **config_by_trial[selected_trial],
            }
            trial_args = build_trial_args(config, args, seed=args.seed + selected_trial)
            row = run_config(
                config,
                trial_args,
                full_train[0],
                full_train[1],
                full_val[0],
                full_val[1],
                x_test,
                y_test,
                target_scaler,
            )
            row["trial"] = f"refit-{label}"
            row["selected_trial"] = selected_trial
            row["config_json"] = json.dumps(compact_config(config), sort_keys=True)
            final_rows.append(row)
            print(
                f"[best refit] {label} selected_trial={selected_trial} "
                f"val_mae={row['best_val_mae']:.6g} test_mae={row['test_mae']:.6g}",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    trials_csv = output_dir / f"jarvis-mace-head-tuning-trials-{stamp}.csv"
    summary_csv = output_dir / f"jarvis-mace-head-tuning-summary-{stamp}.csv"
    json_path = output_dir / f"jarvis-mace-head-tuning-{stamp}.json"
    write_csv(trials_csv, trial_rows)
    write_csv(summary_csv, final_rows or best_by_head)
    payload = {
        "args": vars(args),
        "embedding_file": str(Path(args.embeddings).resolve()),
        "embedding_metadata": loaded["metadata"],
        "n_samples": int(len(y)),
        "feature_dim": int(x.shape[1]),
        "split_sizes": {key: int(len(value)) for key, value in split.items()},
        "tune_split_sizes": {
            "train": int(len(tune_train_idx)),
            "val": int(len(tune_val_idx)),
            "test": int(len(split["test"])),
        },
        "target_scaler": target_scaler.as_dict(),
        "runtime": helpers.runtime_info(device),
        "trials": trial_rows,
        "best_by_head": best_by_head,
        "final_refits": final_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    print(f"Wrote {trials_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
