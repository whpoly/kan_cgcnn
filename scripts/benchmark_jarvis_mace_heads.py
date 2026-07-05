from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgcnn_pyg_kan.kan import make_kan_mlp
from cgcnn_pyg_kan.model import MLPHead


class Standardizer:
    def __init__(self, values: np.ndarray) -> None:
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = np.maximum(values.std(axis=0, keepdims=True), 1e-8)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32, copy=False)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "mean": np.ravel(self.mean).astype(float).tolist(),
            "std": np.ravel(self.std).astype(float).tolist(),
        }


class TargetScaler:
    def __init__(self, values: np.ndarray) -> None:
        self.mean = float(values.mean())
        self.std = float(max(values.std(), 1e-8))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32, copy=False)

    def inverse_transform(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return values * self.std + self.mean

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Linear/MLP/KAN heads on cached JARVIS MACE descriptors."
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument(
        "--heads",
        nargs="+",
        default=["ridge", "linear", "mlp", "kan"],
        choices=["ridge", "linear", "mlp", "kan", "rf"],
    )
    parser.add_argument("--split", default="random", choices=["random", "formula"])
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--kan-hidden-dim", type=int, default=32)
    parser.add_argument("--kan-impl", choices=["spline", "fastkan"], default="fastkan")
    parser.add_argument("--kan-grid-size", type=int, default=3)
    parser.add_argument("--kan-spline-order", type=int, default=3)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-split", type=int, default=2)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-max-features", default="sqrt")
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every-epochs", type=int, default=10)
    parser.add_argument("--output-dir", default="benchmarks/jarvis-mace-heads")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_embeddings(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    ids = np.asarray(data["ids"], dtype=str) if "ids" in data else np.arange(len(y)).astype(str)
    formulas = (
        np.asarray(data["formulas"], dtype=str)
        if "formulas" in data
        else np.asarray(["unknown"] * len(y), dtype=str)
    )
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if not np.all(finite):
        print(f"Filtered {(~finite).sum()} rows with non-finite features/targets.", flush=True)
        x, y, ids, formulas = x[finite], y[finite], ids[finite], formulas[finite]

    metadata = {}
    if "metadata" in data:
        try:
            metadata = json.loads(str(data["metadata"].item()))
        except Exception:
            metadata = {"raw": str(data["metadata"])}
    return {"X": x, "y": y, "ids": ids, "formulas": formulas, "metadata": metadata}


def split_indices(
    n_samples: int,
    formulas: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    if args.test_ratio <= 0 or args.val_ratio < 0 or args.test_ratio + args.val_ratio >= 0.8:
        raise ValueError("Use sensible split ratios: test > 0, val >= 0, and val + test < 0.8")

    indices = np.arange(n_samples)
    if args.split == "random":
        rng = np.random.default_rng(args.seed)
        permuted = rng.permutation(indices)
        test_size = max(1, int(round(n_samples * args.test_ratio)))
        val_size = max(1, int(round(n_samples * args.val_ratio))) if args.val_ratio > 0 else 0
        test_idx = permuted[:test_size]
        val_idx = permuted[test_size : test_size + val_size]
        train_idx = permuted[test_size + val_size :]
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    if args.split == "formula":
        groups = np.asarray(formulas, dtype=str)
        if len(np.unique(groups)) < 3:
            raise ValueError("formula split needs at least three distinct formulas")
        first = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_ratio,
            random_state=args.seed,
        )
        train_val_idx, test_idx = next(first.split(indices, groups=groups))
        if args.val_ratio == 0:
            return {
                "train": indices[train_val_idx],
                "val": np.asarray([], dtype=int),
                "test": indices[test_idx],
            }
        val_ratio_within_train_val = args.val_ratio / (1.0 - args.test_ratio)
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=val_ratio_within_train_val,
            random_state=args.seed + 1,
        )
        train_rel, val_rel = next(
            second.split(train_val_idx, groups=groups[indices[train_val_idx]])
        )
        return {
            "train": indices[train_val_idx][train_rel],
            "val": indices[train_val_idx][val_rel],
            "test": indices[test_idx],
        }

    raise ValueError(f"unsupported split {args.split!r}")


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = float(np.sum((target - target.mean()) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: TargetScaler,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    model = Ridge(alpha=args.ridge_alpha, random_state=args.seed)
    y_train_scaled = target_scaler.transform(y_train)
    model.fit(x_train, y_train_scaled)
    train_seconds = time.perf_counter() - start

    def predict(x: np.ndarray) -> np.ndarray:
        return target_scaler.inverse_transform(model.predict(x)).reshape(-1)

    val_metrics = metrics(predict(x_val), y_val) if len(x_val) else {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    test_metrics = metrics(predict(x_test), y_test)
    train_metrics = metrics(predict(x_train), y_train)
    return {
        "head": "ridge",
        "params": int(model.coef_.size + 1),
        "best_val_mae": val_metrics["mae"],
        "train_mae": train_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "train_seconds": train_seconds,
        "epochs_ran": 0,
    }


def _parse_rf_max_features(value: str | float | int | None) -> str | float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"sqrt", "log2"}:
        return lowered
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            "--rf-max-features must be sqrt, log2, none, or a float fraction"
        ) from exc


def _random_forest_node_count(model: RandomForestRegressor) -> int:
    return int(sum(estimator.tree_.node_count for estimator in model.estimators_))


def fit_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: TargetScaler,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    model = RandomForestRegressor(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_split=args.rf_min_samples_split,
        min_samples_leaf=args.rf_min_samples_leaf,
        max_features=_parse_rf_max_features(args.rf_max_features),
        n_jobs=args.rf_n_jobs,
        random_state=args.seed,
    )
    y_train_scaled = target_scaler.transform(y_train)
    model.fit(x_train, y_train_scaled)
    train_seconds = time.perf_counter() - start

    def predict(x: np.ndarray) -> np.ndarray:
        return target_scaler.inverse_transform(model.predict(x)).reshape(-1)

    val_metrics = (
        metrics(predict(x_val), y_val)
        if len(x_val)
        else {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    )
    train_metrics = metrics(predict(x_train), y_train)
    test_metrics = metrics(predict(x_test), y_test)
    return {
        "head": "rf",
        "params": _random_forest_node_count(model),
        "best_val_mae": val_metrics["mae"],
        "train_mae": train_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "train_seconds": train_seconds,
        "epochs_ran": 0,
    }


def make_torch_head(head: str, in_dim: int, args: argparse.Namespace) -> nn.Module:
    if head == "linear":
        return nn.Linear(in_dim, 1)
    if head == "mlp":
        return MLPHead(in_dim, args.mlp_hidden_dims, out_dim=1, dropout=args.dropout)
    if head == "kan":
        return make_kan_mlp(
            in_dim,
            args.kan_hidden_dim,
            1,
            impl=args.kan_impl,
            dropout=args.dropout,
            grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order,
        )
    raise ValueError(f"unsupported torch head {head!r}")


@torch.no_grad()
def evaluate_torch(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    target_scaler: TargetScaler,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    predictions = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        pred_scaled = model(batch).reshape(-1)
        predictions.append(target_scaler.inverse_transform(pred_scaled).cpu().numpy())
    return metrics(np.concatenate(predictions), y)


def fit_torch_head(
    head: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: TargetScaler,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    device = torch.device(args.device)
    model = make_torch_head(head, x_train.shape[1], args).to(device)
    y_train_scaled = target_scaler.transform(y_train)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_scaled))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_mae = float("inf")
    best_epoch = 0
    no_improve = 0
    start_time = time.perf_counter()
    epochs_ran = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch).reshape_as(y_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(x_batch)
            total_rows += len(x_batch)

        epochs_ran = epoch
        if len(x_val):
            val_metrics = evaluate_torch(
                model,
                x_val,
                y_val,
                target_scaler,
                device,
                args.batch_size,
            )
            val_mae = val_metrics["mae"]
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if args.log_every_epochs > 0 and (
                epoch % args.log_every_epochs == 0 or epoch == 1
            ):
                train_loss = total_loss / max(total_rows, 1)
                print(
                    f"[{head}] epoch {epoch}/{args.epochs} "
                    f"train_loss={train_loss:.6g} val_mae={val_mae:.6g} "
                    f"best={best_val_mae:.6g}@{best_epoch}",
                    flush=True,
                )
            if args.patience > 0 and no_improve >= args.patience:
                break
        elif args.log_every_epochs > 0 and (epoch % args.log_every_epochs == 0 or epoch == 1):
            train_loss = total_loss / max(total_rows, 1)
            print(f"[{head}] epoch {epoch}/{args.epochs} train_loss={train_loss:.6g}", flush=True)

    train_seconds = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val_mae = float("nan")

    train_metrics = evaluate_torch(model, x_train, y_train, target_scaler, device, args.batch_size)
    test_metrics = evaluate_torch(model, x_test, y_test, target_scaler, device, args.batch_size)
    return {
        "head": head,
        "params": count_parameters(model),
        "best_val_mae": best_val_mae,
        "train_mae": train_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "train_seconds": train_seconds,
        "epochs_ran": epochs_ran,
    }


def print_table(rows: Iterable[dict[str, float | int | str]]) -> None:
    rows = list(rows)
    headers = [
        "head",
        "params",
        "best_val_mae",
        "train_mae",
        "test_mae",
        "test_rmse",
        "test_r2",
        "train_seconds",
        "epochs_ran",
    ]
    widths = {
        header: max(len(header), *(len(format_value(row[header])) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(format_value(row[header]).ljust(widths[header]) for header in headers))


def format_value(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def runtime_info(device: torch.device) -> dict[str, object]:
    cuda_index = device.index if device.index is not None else 0
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(cuda_index)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    loaded = load_embeddings(Path(args.embeddings))
    x = loaded["X"]
    y = loaded["y"]
    formulas = loaded["formulas"]
    split = split_indices(len(y), formulas, args)

    feature_scaler = Standardizer(x[split["train"]])
    target_scaler = TargetScaler(y[split["train"]])
    x_scaled = feature_scaler.transform(x)

    x_train, y_train = x_scaled[split["train"]], y[split["train"]]
    x_val, y_val = x_scaled[split["val"]], y[split["val"]]
    x_test, y_test = x_scaled[split["test"]], y[split["test"]]

    print(
        f"Loaded {len(y)} embeddings, feature_dim={x.shape[1]}, "
        f"split train/val/test={len(y_train)}/{len(y_val)}/{len(y_test)}, device={device}",
        flush=True,
    )

    rows: list[dict[str, float | int | str]] = []
    for head in args.heads:
        if head == "ridge":
            row = fit_ridge(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                target_scaler,
                args,
            )
        elif head == "rf":
            row = fit_random_forest(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                target_scaler,
                args,
            )
        else:
            row = fit_torch_head(
                head,
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                target_scaler,
                args,
            )
        rows.append(row)

    print_table(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"jarvis-mace-heads-{stamp}.csv"
    json_path = output_dir / f"jarvis-mace-heads-{stamp}.json"
    payload = {
        "args": vars(args),
        "embedding_file": str(Path(args.embeddings).resolve()),
        "embedding_metadata": loaded["metadata"],
        "n_samples": int(len(y)),
        "feature_dim": int(x.shape[1]),
        "split": {key: value.astype(int).tolist() for key, value in split.items()},
        "split_sizes": {key: int(len(value)) for key, value in split.items()},
        "target_scaler": target_scaler.as_dict(),
        "runtime": runtime_info(device),
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
