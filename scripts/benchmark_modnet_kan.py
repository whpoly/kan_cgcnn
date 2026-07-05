from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgcnn_pyg_kan.kan import FastKANLinear, KANLinear
from cgcnn_pyg_kan.modnet import MODNetKAN
from cgcnn_pyg_kan.modnet_features import MODNetFeatureProcessor, make_feature_frame

FEATURE_PRESETS = [
    "auto",
    "pymatgen-composition",
    "pymatgen-structure",
    "matminer-composition",
    "matminer-structure-lite",
]
MODEL_CHOICES = ["mlp", "fastkan", "spline", "kan"]
TASK_TYPES = ("regression", "classification")
METRIC_NAMES = [
    "mae",
    "rmse",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "rocauc",
]


class TargetScaler:
    def __init__(self, values: np.ndarray, mode: str = "none") -> None:
        values = np.asarray(values, dtype=np.float32)
        self.mode = mode
        if mode == "none":
            self.mean = 0.0
            self.std = 1.0
        elif mode == "standard":
            self.mean = float(values.mean())
            self.std = float(max(values.std(), 1e-8))
        else:
            raise ValueError("target scaler mode must be 'none' or 'standard'")

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32, copy=False)

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def as_dict(self) -> dict[str, float]:
        return {"mode": self.mode, "mean": self.mean, "std": self.std}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MODNet-style KAN model on Matbench descriptor features."
    )
    parser.add_argument("--dataset", default="matbench_phonons")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=["fastkan"])
    parser.add_argument("--featurizer-preset", choices=FEATURE_PRESETS, default="auto")
    parser.add_argument("--featurizer-jobs", type=int, default=1)
    parser.add_argument(
        "--precomputed-feature-dir",
        default=None,
        help=(
            "Directory containing official MODNet-style fold exports. When set, "
            "fold_<n>/train_features.csv.gz and test_features.csv.gz are used "
            "instead of refitting local matminer features and feature selection."
        ),
    )
    parser.add_argument("--n-features", type=int, default=256)
    parser.add_argument("--common-dims", type=int, nargs="+", default=[64])
    parser.add_argument("--group-dims", type=int, nargs="+", default=[32])
    parser.add_argument("--property-dims", type=int, nargs="+", default=[16])
    parser.add_argument("--target-dims", type=int, nargs="*", default=[])
    parser.add_argument("--kan-impl", choices=["fastkan", "spline"], default="fastkan")
    parser.add_argument("--kan-grid-size", type=int, default=5)
    parser.add_argument("--kan-spline-order", type=int, default=3)
    parser.add_argument("--scaler", choices=["minmax", "standard", "none"], default="minmax")
    parser.add_argument("--target-scale", choices=["none", "standard"], default="none")
    parser.add_argument("--impute-strategy", default="median")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--loss",
        choices=["auto", "mae", "rmse", "mse", "bce"],
        default="auto",
        help=(
            "Training objective. auto uses mae for regression and bce for classification. "
            "Regression also supports rmse; mse is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--prune-kan-fraction",
        type=float,
        default=0.0,
        help=(
            "Global magnitude-prune this fraction of trainable KAN-family "
            "parameters after training and before validation/test reporting."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help=(
            "Do not load, featurize, or evaluate the official Matbench test fold. "
            "Use this for hyperparameter tuning; final Matbench records require test evaluation."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--log-every-epochs", type=int, default=10)
    parser.add_argument("--forward-iters", type=int, default=40)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument(
        "--export-formulas",
        action="store_true",
        help="Write sparse layerwise formula text files for KAN-family models after pruning.",
    )
    parser.add_argument(
        "--formula-top-k",
        type=int,
        default=40,
        help="Maximum terms per layer/output branch in exported formula files; 0 writes all nonzero terms.",
    )
    parser.add_argument(
        "--formula-min-abs",
        type=float,
        default=0.0,
        help="Minimum absolute coefficient to include in exported formula files.",
    )
    parser.add_argument(
        "--no-matbench-records",
        action="store_true",
        help="Do not write MatbenchTask records even when running full official test folds.",
    )
    return parser.parse_args()


def normalize_loss_name(loss_name: str) -> str:
    return "rmse" if loss_name == "mse" else loss_name


def validate_task_loss(task_type: str, loss_name: str) -> str:
    if loss_name == "auto":
        return "bce" if task_type == "classification" else "mae"
    loss_name = normalize_loss_name(loss_name)
    if task_type == "regression" and loss_name not in ("mae", "rmse"):
        raise ValueError(f"Regression tasks support --loss mae or rmse, got {loss_name!r}")
    if task_type == "classification" and loss_name != "bce":
        raise ValueError(f"Classification tasks require --loss bce, got {loss_name!r}")
    return loss_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was set, but --device is not a CUDA device")
    return device


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def runtime_info(device: torch.device) -> dict[str, str | int | bool | None]:
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
        "cuda_device_count": torch.cuda.device_count(),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_nonzero_parameters(model: torch.nn.Module) -> int:
    return int(
        sum(
            torch.count_nonzero(parameter.detach()).item()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def apply_global_magnitude_pruning(model: torch.nn.Module, fraction: float) -> int:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("--prune-kan-fraction must be in [0, 1)")
    if fraction <= 0:
        return 0

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total = sum(parameter.numel() for parameter in params)
    n_prune = int(round(total * fraction))
    if n_prune <= 0:
        return 0

    scores = torch.cat([parameter.detach().abs().flatten().cpu() for parameter in params])
    prune_indices = torch.topk(scores, k=min(n_prune, total - 1), largest=False).indices
    offset = 0
    pruned = 0
    for parameter in params:
        next_offset = offset + parameter.numel()
        local = prune_indices[(prune_indices >= offset) & (prune_indices < next_offset)] - offset
        if local.numel() > 0:
            flat = parameter.data.view(-1)
            flat[local.to(device=flat.device)] = 0
            pruned += int(local.numel())
        offset = next_offset
    return pruned


def adamw_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower_name = name.lower()
        if (
            weight_decay == 0.0
            or parameter.ndim < 2
            or lower_name.endswith(".bias")
            or "norm" in lower_name
        ):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    groups: list[dict[str, object]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def select_subset(items, targets, size: int | None, seed: int):
    if size is None or size >= len(items):
        return list(items), np.asarray(targets, dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(items), size=size, replace=False)
    return [items.iloc[i] for i in indices], np.asarray([targets.iloc[i] for i in indices], dtype=np.float32)


def load_matbench_split(args: argparse.Namespace, fold: int) -> dict[str, Any]:
    from matbench.metadata import mbv01_metadata
    from matbench.task import MatbenchTask

    metadata = mbv01_metadata[args.dataset]
    if metadata.input_type not in ("structure", "composition"):
        raise ValueError(
            f"{args.dataset} input_type is {metadata.input_type!r}; expected structure or composition"
        )
    if metadata.task_type not in TASK_TYPES:
        raise ValueError(
            f"{args.dataset} task_type is {metadata.task_type!r}; expected one of {TASK_TYPES}"
        )
    args.loss = validate_task_loss(metadata.task_type, args.loss)
    if metadata.task_type == "classification" and args.target_scale != "none":
        raise ValueError("Classification tasks require --target-scale none")

    task = MatbenchTask(args.dataset, autoload=False)
    task.load()
    train_inputs, train_targets = task.get_train_and_val_data(fold)
    train_inputs, train_targets = select_subset(train_inputs, train_targets, args.train_size, args.seed)
    if args.skip_test_eval:
        fold_key = task.folds_map[fold]
        official_test_size = len(task.validation[fold_key].test)
        test_inputs = None
        test_targets = np.array([], dtype=np.float32)
        test_size = 0
    else:
        test_inputs, test_targets = task.get_test_data(fold, include_target=True)
        test_inputs, test_targets = select_subset(test_inputs, test_targets, args.test_size, args.seed + 1)
        official_test_size = len(test_targets)
        test_size = len(test_inputs)

    return {
        "metadata": {
            "dataset": args.dataset,
            "fold": fold,
            "task_type": metadata.task_type,
            "target": metadata.target,
            "input_type": metadata.input_type,
            "unit": getattr(metadata, "unit", None),
            "mad": getattr(metadata, "mad", None),
            "frac_true": getattr(metadata, "frac_true", None),
            "n_samples": metadata.n_samples,
            "train_and_val_size": len(train_inputs),
            "test_size": test_size,
            "official_test_size": official_test_size,
        },
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "test_inputs": test_inputs,
        "test_targets": test_targets,
    }


def find_precomputed_fold_dir(base_dir: Path, fold: int) -> Path:
    candidates = [
        base_dir / f"fold_{fold}",
        base_dir / f"fold{fold}",
        base_dir / f"fold-{fold}",
    ]
    if base_dir.name in {f"fold_{fold}", f"fold{fold}", f"fold-{fold}"}:
        candidates.insert(0, base_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find precomputed feature directory for fold {fold} under {base_dir}"
    )


def read_first_existing_csv(paths: Iterable[Path]) -> Any:
    for path in paths:
        if path.exists():
            return path, __import__("pandas").read_csv(path, index_col=0)
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of these files exists: {joined}")


def load_precomputed_feature_split(args: argparse.Namespace, fold: int) -> dict[str, Any]:
    import pandas as pd

    from matbench.metadata import mbv01_metadata

    if args.precomputed_feature_dir is None:
        raise ValueError("--precomputed-feature-dir was not provided")
    metadata = mbv01_metadata[args.dataset]
    if metadata.task_type not in TASK_TYPES:
        raise ValueError(
            f"{args.dataset} task_type is {metadata.task_type!r}; expected one of {TASK_TYPES}"
        )
    args.loss = validate_task_loss(metadata.task_type, args.loss)
    if metadata.task_type == "classification" and args.target_scale != "none":
        raise ValueError("Classification tasks require --target-scale none")

    fold_dir = find_precomputed_fold_dir(Path(args.precomputed_feature_dir), fold)
    _, train_features = read_first_existing_csv(
        [
            fold_dir / "train_features.csv.gz",
            fold_dir / "train_features.csv",
            fold_dir / f"train_features_f{fold}.csv.gz",
            fold_dir / f"train_features_f{fold}.csv",
        ]
    )
    _, test_features = read_first_existing_csv(
        [
            fold_dir / "test_features.csv.gz",
            fold_dir / "test_features.csv",
            fold_dir / f"test_features_f{fold}.csv.gz",
            fold_dir / f"test_features_f{fold}.csv",
        ]
    )
    _, train_targets_frame = read_first_existing_csv(
        [
            fold_dir / "train_targets.csv.gz",
            fold_dir / "train_targets.csv",
            fold_dir / f"train_targets_f{fold}.csv.gz",
            fold_dir / f"train_targets_f{fold}.csv",
        ]
    )
    _, test_targets_frame = read_first_existing_csv(
        [
            fold_dir / "test_targets.csv.gz",
            fold_dir / "test_targets.csv",
            fold_dir / f"test_targets_f{fold}.csv.gz",
            fold_dir / f"test_targets_f{fold}.csv",
        ]
    )
    target_column = str(metadata.target)
    if target_column not in train_targets_frame.columns:
        target_column = str(train_targets_frame.columns[0])
    if target_column not in test_targets_frame.columns:
        target_column = str(test_targets_frame.columns[0])

    feature_order_path = fold_dir / "feature_order.json"
    if feature_order_path.exists():
        feature_order = json.loads(feature_order_path.read_text(encoding="utf-8-sig"))
        feature_order = [feature for feature in feature_order if feature in train_features.columns]
        train_features = train_features.reindex(columns=feature_order)
        test_features = test_features.reindex(columns=feature_order)

    train_features = train_features.apply(pd.to_numeric, errors="coerce")
    test_features = test_features.apply(pd.to_numeric, errors="coerce")
    return {
        "metadata": {
            "dataset": args.dataset,
            "fold": fold,
            "task_type": metadata.task_type,
            "target": target_column,
            "input_type": metadata.input_type,
            "unit": getattr(metadata, "unit", None),
            "mad": getattr(metadata, "mad", None),
            "frac_true": getattr(metadata, "frac_true", None),
            "n_samples": metadata.n_samples,
            "train_and_val_size": len(train_features),
            "test_size": len(test_features),
            "official_test_size": len(test_features),
            "precomputed_feature_dir": str(fold_dir),
        },
        "precomputed_features": True,
        "train_features": train_features,
        "test_features": test_features,
        "train_targets": train_targets_frame[target_column].to_numpy(dtype=np.float32),
        "test_targets": test_targets_frame[target_column].to_numpy(dtype=np.float32),
    }


def resolve_feature_preset(input_type: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if input_type == "structure":
        return "matminer-structure-lite"
    return "matminer-composition"


def featurize_with_fallback(
    materials: list,
    preset: str,
    input_type: str,
    n_jobs: int,
) -> tuple[Any, str]:
    try:
        return make_feature_frame(materials, preset=preset, n_jobs=n_jobs), preset
    except Exception as exc:
        fallback = "pymatgen-structure" if input_type == "structure" else "pymatgen-composition"
        print(
            f"Feature preset {preset!r} failed with {exc.__class__.__name__}: {exc}. "
            f"Falling back to {fallback!r}.",
            flush=True,
        )
        return make_feature_frame(materials, preset=fallback, n_jobs=n_jobs), fallback


def split_train_val(
    size: int,
    val_ratio: float,
    seed: int,
    targets: np.ndarray | None = None,
    task_type: str = "regression",
) -> tuple[np.ndarray, np.ndarray]:
    if val_ratio == 0.0:
        return np.arange(size), np.array([], dtype=int)
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val_ratio must be 0 or in (0, 0.5)")
    if task_type == "classification" and targets is not None:
        labels = np.asarray(targets).astype(int)
        if len(np.unique(labels)) >= 2 and np.bincount(labels).min() >= 2:
            from sklearn.model_selection import StratifiedShuffleSplit

            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=max(1, int(round(size * val_ratio))),
                random_state=seed,
            )
            train_indices, val_indices = next(splitter.split(np.zeros(size), labels))
            return train_indices, val_indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(size)
    val_size = max(1, int(round(size * val_ratio)))
    return indices[val_size:], indices[:val_size]


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model(
    model_name: str,
    n_feat: int,
    target_name: str,
    args: argparse.Namespace,
) -> MODNetKAN:
    family = canonical_model_name(model_name, args)
    return MODNetKAN(
        n_feat=n_feat,
        targets=[[[target_name]]],
        num_neurons=(
            args.common_dims,
            args.group_dims,
            args.property_dims,
            args.target_dims,
        ),
        block_type="mlp" if family == "mlp" else "kan",
        kan_impl="spline" if family == "spline" else "fastkan",
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
        dropout=args.dropout,
    )


def canonical_model_name(model_name: str, args: argparse.Namespace) -> str:
    if model_name == "kan":
        return args.kan_impl
    return model_name


def regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    if loss_name == "mae":
        return F.l1_loss(prediction, target)
    if loss_name in ("rmse", "mse"):
        return F.mse_loss(prediction, target)
    raise ValueError(f"unsupported loss {loss_name!r}")


def task_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task_type: str,
    loss_name: str,
) -> torch.Tensor:
    if task_type == "classification":
        return F.binary_cross_entropy_with_logits(prediction, target)
    return regression_loss(prediction, target, loss_name)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: str,
    loss_name: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch_x)
        loss = task_loss(prediction.view_as(batch_y), batch_y, task_type, loss_name)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch_y)
        total_samples += len(batch_y)
    return total_loss / total_samples


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
    task_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds = []
    targets = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        raw_prediction = model(batch_x).view_as(batch_y)
        if task_type == "classification":
            prediction = torch.sigmoid(raw_prediction)
            target = batch_y
        else:
            prediction = scaler.inverse_transform_tensor(raw_prediction)
            target = scaler.inverse_transform_tensor(batch_y)
        preds.append(prediction.cpu())
        targets.append(target.cpu())
    if not preds:
        return torch.empty(0), torch.empty(0)
    return torch.cat(preds), torch.cat(targets)


def metrics_from_predictions(
    pred: torch.Tensor,
    target: torch.Tensor,
    task_type: str,
) -> dict[str, float]:
    if len(pred) == 0:
        return {name: float("nan") for name in METRIC_NAMES}
    if task_type == "classification":
        y_true = target.numpy().astype(int).reshape(-1)
        y_prob = pred.numpy().reshape(-1)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {name: float("nan") for name in METRIC_NAMES}
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            metrics["rocauc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["rocauc"] = float("nan")
        return metrics

    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(F.mse_loss(pred, target)).item()
    metrics = {name: float("nan") for name in METRIC_NAMES}
    metrics["mae"] = mae
    metrics["rmse"] = rmse
    return metrics


def primary_val_metric(task_type: str) -> str:
    return "rocauc" if task_type == "classification" else "mae"


def metric_is_better(candidate: float, incumbent: float, task_type: str) -> bool:
    if not np.isfinite(candidate):
        return False
    if not np.isfinite(incumbent):
        return True
    return candidate > incumbent if task_type == "classification" else candidate < incumbent


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
    task_type: str,
) -> dict[str, float]:
    pred, target = predict_loader(model, loader, scaler, device, task_type)
    return metrics_from_predictions(pred, target, task_type)


@torch.no_grad()
def benchmark_forward(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    warmup_iters: int,
    forward_iters: int,
) -> float:
    model.eval()
    batch = torch.from_numpy(x[: min(len(x), 256)]).to(device)
    for _ in range(warmup_iters):
        _ = model(batch)
    sync(device)
    start = time.perf_counter()
    for _ in range(forward_iters):
        _ = model(batch)
    sync(device)
    return 1000.0 * (time.perf_counter() - start) / forward_iters


def run_model(
    model_name: str,
    fold_data: dict[str, Any],
    prepared: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float | int | str], list[float]]:
    set_seed(args.seed)
    model_label = canonical_model_name(model_name, args)
    task_type = str(fold_data["metadata"]["task_type"])
    target_name = str(fold_data["metadata"]["target"])
    model = build_model(
        model_name,
        n_feat=prepared["x_train"].shape[1],
        target_name=target_name,
        args=args,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model, args.weight_decay),
        lr=args.lr,
    )
    train_loader = make_loader(
        prepared["x_train"],
        prepared["y_train_scaled"],
        args.batch_size,
        shuffle=True,
    )
    val_loader = (
        make_loader(prepared["x_val"], prepared["y_val_scaled"], args.batch_size, shuffle=False)
        if len(prepared["x_val"]) > 0
        else None
    )
    test_loader = (
        make_loader(prepared["x_test"], prepared["y_test_scaled"], args.batch_size, shuffle=False)
        if not args.skip_test_eval
        else None
    )

    best_state = None
    best_metric = float("nan")
    best_val_metrics = {name: float("nan") for name in METRIC_NAMES}
    val_metric_name = primary_val_metric(task_type)
    best_epoch = 0
    patience_left = args.early_stopping_patience
    print(
        f"\n[{model_label}] fold {fold_data['metadata']['fold']} start: "
        f"features={prepared['x_train'].shape[1]}, epochs={args.epochs}",
        flush=True,
    )
    sync(device)
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, task_type, args.loss)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, prepared["target_scaler"], device, task_type)
            improved = metric_is_better(val_metrics[val_metric_name], best_metric, task_type)
            if improved:
                best_metric = val_metrics[val_metric_name]
                best_val_metrics = val_metrics
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                }
                patience_left = args.early_stopping_patience
            else:
                patience_left -= 1

            if args.log_every_epochs > 0 and (
                epoch % args.log_every_epochs == 0 or epoch == 1 or epoch == args.epochs
            ):
                print(
                    f"[{model_label}] epoch {epoch}/{args.epochs} "
                    f"train_loss={train_loss:.6g} "
                    f"val_{val_metric_name}={val_metrics[val_metric_name]:.6g} "
                    f"best_val_{val_metric_name}={best_metric:.6g}",
                    flush=True,
                )
            if args.early_stopping_patience > 0 and patience_left <= 0:
                print(f"[{model_label}] early stop at epoch {epoch}", flush=True)
                break
        elif args.log_every_epochs > 0 and (
            epoch % args.log_every_epochs == 0 or epoch == 1 or epoch == args.epochs
        ):
            print(f"[{model_label}] epoch {epoch}/{args.epochs} train_loss={train_loss:.6g}", flush=True)
    sync(device)
    train_seconds = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)
    params = count_parameters(model)
    pruned_params = 0
    if model_label in ("fastkan", "spline") and args.prune_kan_fraction > 0:
        pruned_params = apply_global_magnitude_pruning(model, args.prune_kan_fraction)
        if val_loader is not None:
            best_val_metrics = evaluate(model, val_loader, prepared["target_scaler"], device, task_type)
    if test_loader is None:
        test_prediction = torch.empty(0)
        test_metrics = {name: float("nan") for name in METRIC_NAMES}
        forward_features = prepared["x_val"] if len(prepared["x_val"]) else prepared["x_train"]
    else:
        test_prediction, test_target = predict_loader(
            model,
            test_loader,
            prepared["target_scaler"],
            device,
            task_type,
        )
        test_metrics = metrics_from_predictions(test_prediction, test_target, task_type)
        forward_features = prepared["x_test"]
    forward_ms = benchmark_forward(
        model,
        forward_features,
        device,
        warmup_iters=args.warmup_iters,
        forward_iters=args.forward_iters,
    )
    effective_params = count_nonzero_parameters(model)
    row = {
        "fold": fold_data["metadata"]["fold"],
        "task_type": task_type,
        "model": model_label,
        "block_type": "mlp" if model_label == "mlp" else "kan",
        "kan_impl": model_label if model_label in ("fastkan", "spline") else "none",
        "kan_grid_size": args.kan_grid_size if model_label in ("fastkan", "spline") else "none",
        "kan_spline_order": args.kan_spline_order if model_label == "spline" else "none",
        "featurizer_preset": prepared["feature_preset"],
        "n_features": prepared["x_train"].shape[1],
        "params": params,
        "effective_params": effective_params,
        "pruned_params": pruned_params,
        "params_before_prune": params,
        "params_after_prune": effective_params,
        "params_pruned": pruned_params,
        "params_pruned_pct": 100.0 * pruned_params / params if params else 0.0,
        "prune_kan_fraction": args.prune_kan_fraction if model_label in ("fastkan", "spline") else 0.0,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "forward_ms_per_batch": forward_ms,
    }
    for metric_name in METRIC_NAMES:
        row[f"best_val_{metric_name}"] = best_val_metrics[metric_name]
        row[f"test_{metric_name}"] = test_metrics[metric_name]
    if args.export_formulas and model_label in ("fastkan", "spline"):
        formula_path = (
            Path(args.output_dir)
            / f"formula-{args.dataset}-fold{fold_data['metadata']['fold']}-{model_label}.txt"
        )
        export_sparse_formula(
            model,
            model_label=model_label,
            input_names=prepared["selected_features"],
            output_path=formula_path,
            top_k=args.formula_top_k,
            min_abs=args.formula_min_abs,
        )
        row["formula_path"] = str(formula_path)
    return row, [float(value) for value in test_prediction.numpy().reshape(-1)]


def prepare_fold(
    fold_data: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if fold_data.get("precomputed_features"):
        return prepare_precomputed_fold(fold_data, args)

    requested_preset = resolve_feature_preset(
        fold_data["metadata"]["input_type"],
        args.featurizer_preset,
    )
    start = time.perf_counter()
    train_features, actual_train_preset = featurize_with_fallback(
        fold_data["train_inputs"],
        requested_preset,
        fold_data["metadata"]["input_type"],
        args.featurizer_jobs,
    )
    test_features = None
    if not args.skip_test_eval:
        test_features, actual_test_preset = featurize_with_fallback(
            fold_data["test_inputs"],
            actual_train_preset,
            fold_data["metadata"]["input_type"],
            args.featurizer_jobs,
        )
        if actual_test_preset != actual_train_preset:
            raise RuntimeError("train and test featurization used different presets")
    featurize_seconds = time.perf_counter() - start

    train_indices, val_indices = split_train_val(
        len(train_features),
        args.val_ratio,
        args.seed,
        targets=fold_data["train_targets"],
        task_type=fold_data["metadata"]["task_type"],
    )
    train_targets = fold_data["train_targets"]
    test_targets = fold_data["test_targets"]
    processor = MODNetFeatureProcessor(
        n_features=args.n_features,
        scaler=args.scaler,
        impute_strategy=args.impute_strategy,
        random_state=args.seed,
        task_type=fold_data["metadata"]["task_type"],
    )
    processor.fit(train_features.iloc[train_indices], train_targets[train_indices])

    x_all_train = processor.transform(train_features)
    x_test = (
        processor.transform(test_features)
        if test_features is not None
        else np.empty((0, x_all_train.shape[1]), dtype=np.float32)
    )
    y_train = train_targets[train_indices]
    y_val = train_targets[val_indices] if len(val_indices) else np.array([], dtype=np.float32)
    target_scaler = TargetScaler(y_train, mode=args.target_scale)
    y_all_train_scaled = target_scaler.transform(train_targets)
    y_test_scaled = (
        target_scaler.transform(test_targets)
        if len(test_targets)
        else np.array([], dtype=np.float32)
    )

    return {
        "feature_preset": actual_train_preset,
        "featurize_seconds": featurize_seconds,
        "processor": processor,
        "selected_features": processor.selected_columns_,
        "target_scaler": target_scaler,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_targets),
        "x_train": x_all_train[train_indices],
        "y_train_scaled": y_all_train_scaled[train_indices],
        "x_val": x_all_train[val_indices] if len(val_indices) else np.empty((0, x_all_train.shape[1]), dtype=np.float32),
        "y_val_scaled": y_all_train_scaled[val_indices] if len(val_indices) else np.array([], dtype=np.float32),
        "x_test": x_test,
        "y_test_scaled": y_test_scaled,
    }


def make_feature_pipeline(args: argparse.Namespace) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy=args.impute_strategy))]
    if args.scaler == "minmax":
        steps.append(("scaler", MinMaxScaler(feature_range=(-0.5, 0.5))))
    elif args.scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif args.scaler != "none":
        raise ValueError("scaler must be 'minmax', 'standard', or 'none'")
    return Pipeline(steps)


def prepare_precomputed_fold(
    fold_data: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_features = fold_data["train_features"]
    test_features = fold_data["test_features"]
    if args.n_features < 1:
        raise ValueError("--n-features must be at least 1")
    selected_features = list(train_features.columns[: min(args.n_features, train_features.shape[1])])
    if not selected_features:
        raise ValueError("precomputed feature fold contains no usable feature columns")

    train_indices, val_indices = split_train_val(
        len(train_features),
        args.val_ratio,
        args.seed,
        targets=fold_data["train_targets"],
        task_type=fold_data["metadata"]["task_type"],
    )
    train_targets = np.asarray(fold_data["train_targets"], dtype=np.float32)
    test_targets = np.asarray(fold_data["test_targets"], dtype=np.float32)

    pipeline = make_feature_pipeline(args)
    pipeline.fit(train_features.iloc[train_indices][selected_features])
    x_all_train = pipeline.transform(train_features[selected_features]).astype(np.float32, copy=False)
    x_test = pipeline.transform(test_features.reindex(columns=selected_features)).astype(np.float32, copy=False)

    y_train = train_targets[train_indices]
    y_val = train_targets[val_indices] if len(val_indices) else np.array([], dtype=np.float32)
    target_scaler = TargetScaler(y_train, mode=args.target_scale)
    y_all_train_scaled = target_scaler.transform(train_targets)
    y_test_scaled = target_scaler.transform(test_targets) if len(test_targets) else np.array([], dtype=np.float32)

    return {
        "feature_preset": "official-modnet-precomputed",
        "featurize_seconds": 0.0,
        "processor": None,
        "selected_features": selected_features,
        "target_scaler": target_scaler,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_targets),
        "x_train": x_all_train[train_indices],
        "y_train_scaled": y_all_train_scaled[train_indices],
        "x_val": x_all_train[val_indices] if len(val_indices) else np.empty((0, x_all_train.shape[1]), dtype=np.float32),
        "y_val_scaled": y_all_train_scaled[val_indices] if len(val_indices) else np.array([], dtype=np.float32),
        "x_test": x_test,
        "y_test_scaled": y_test_scaled,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    summary = []
    for model_name, model_rows in grouped.items():
        item: dict[str, Any] = {
            "model": model_name,
            "task_type": model_rows[0].get("task_type", ""),
            "folds": len(model_rows),
            "params_mean": float(np.mean([float(row["params"]) for row in model_rows])),
            "effective_params_mean": float(np.mean([float(row["effective_params"]) for row in model_rows])),
            "effective_params_std": finite_std([float(row["effective_params"]) for row in model_rows]),
            "pruned_params_mean": float(np.mean([float(row["pruned_params"]) for row in model_rows])),
            "train_seconds_mean": float(np.mean([float(row["train_seconds"]) for row in model_rows])),
        }
        formula_paths = [str(row["formula_path"]) for row in model_rows if row.get("formula_path")]
        if formula_paths:
            item["formula_paths"] = ";".join(formula_paths)
        item["params_before_prune_mean"] = item["params_mean"]
        item["params_after_prune_mean"] = item["effective_params_mean"]
        item["params_after_prune_std"] = item["effective_params_std"]
        item["params_pruned_mean"] = item["pruned_params_mean"]
        item["params_pruned_pct_mean"] = (
            100.0 * item["params_pruned_mean"] / item["params_before_prune_mean"]
            if item["params_before_prune_mean"]
            else 0.0
        )
        for metric_name in METRIC_NAMES:
            for prefix in ("best_val", "test"):
                key = f"{prefix}_{metric_name}"
                item[f"{key}_mean"] = finite_mean([float(row[key]) for row in model_rows])
                item[f"{key}_std"] = finite_std([float(row[key]) for row in model_rows])
        summary.append(item)
    return summary


def finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else float("nan")


def finite_std(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    return float(np.std(finite_values)) if finite_values else float("nan")


def export_sparse_formula(
    model: torch.nn.Module,
    model_label: str,
    input_names: list[str] | None,
    output_path: Path,
    top_k: int,
    min_abs: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Sparse formula summary for {model_label}",
        "The model is kept in layerwise form so the expression stays readable.",
        "params_after_prune counts nonzero trainable parameters after pruning.",
        "",
        "FastKAN layer form:",
        "  y_o = sum_i a[o,i] * SiLU(x_i) + b_o",
        "        + sum_{i,k} c[o,i,k] * exp(-((LN(x)_i - grid_k) / h)^2)",
        "B-spline KAN layer form:",
        "  y_o = sum_i a[o,i] * SiLU(x_i) + sum_{i,k} c[o,i,k] * B_{i,k}(x_i)",
        "",
    ]
    names_by_module: dict[str, list[str] | None] = {"": input_names}
    for name, module in model.named_modules():
        if isinstance(module, FastKANLinear):
            source_names = source_names_for(name, names_by_module, input_names)
            lines.extend(_fastkan_formula_lines(name, module, source_names, top_k, min_abs))
            remember_output_names(name, module.out_features, names_by_module)
        elif isinstance(module, KANLinear):
            source_names = source_names_for(name, names_by_module, input_names)
            lines.extend(_spline_formula_lines(name, module, source_names, top_k, min_abs))
            remember_output_names(name, module.out_features, names_by_module)
        elif isinstance(module, torch.nn.Linear) and name.startswith("output_heads."):
            source_names = source_names_for(name, names_by_module, input_names)
            lines.extend(_linear_formula_lines(name, module, source_names, top_k, min_abs))
            remember_output_names(name, module.out_features, names_by_module)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parent_name(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def source_names_for(
    name: str,
    names_by_module: dict[str, list[str] | None],
    input_names: list[str] | None,
) -> list[str] | None:
    if name.startswith("output_heads."):
        prop_key = name.split(".", 1)[1]
        return (
            names_by_module.get(f"target_blocks.{prop_key}")
            or names_by_module.get(f"property_blocks.{prop_key}")
            or names_by_module.get(parent_name(name))
        )
    if name.startswith("common_block."):
        return names_by_module.get(parent_name(name)) or input_names
    if name.startswith("group_blocks."):
        return names_by_module.get(parent_name(name)) or names_by_module.get("common_block")
    if name.startswith("property_blocks."):
        prop_key = name.split(".")[1]
        group_key = prop_key.split("_p", 1)[0]
        return (
            names_by_module.get(parent_name(name))
            or names_by_module.get(f"group_blocks.{group_key}")
        )
    if name.startswith("target_blocks."):
        prop_key = name.split(".")[1]
        return (
            names_by_module.get(parent_name(name))
            or names_by_module.get(f"property_blocks.{prop_key}")
        )
    return names_by_module.get(parent_name(name))


def remember_output_names(
    name: str,
    out_features: int,
    names_by_module: dict[str, list[str] | None],
) -> None:
    names = [f"{name}.y{idx}" for idx in range(out_features)]
    names_by_module[name] = names
    parent = parent_name(name)
    if parent:
        names_by_module[parent] = names
    for prefix in ("common_block", "group_blocks", "property_blocks", "target_blocks"):
        if name.startswith(prefix + "."):
            parts = name.split(".")
            if prefix == "common_block":
                names_by_module["common_block"] = names
            elif len(parts) >= 2:
                names_by_module[".".join(parts[:2])] = names


def _source_name(source_names: list[str] | None, index: int) -> str:
    if source_names is not None and index < len(source_names):
        return _safe_symbol(source_names[index])
    return f"x{index}"


def _safe_symbol(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


def _selected_terms(
    coefficients: list[tuple[float, str]],
    top_k: int,
    min_abs: float,
) -> list[tuple[float, str]]:
    filtered = [(coef, text) for coef, text in coefficients if abs(coef) > min_abs]
    filtered.sort(key=lambda item: abs(item[0]), reverse=True)
    return filtered if top_k <= 0 else filtered[:top_k]


def _format_terms(terms: list[tuple[float, str]]) -> str:
    if not terms:
        return "0"
    return " + ".join(f"{coef:.6g}*{text}" for coef, text in terms)


def _fastkan_formula_lines(
    name: str,
    module: FastKANLinear,
    source_names: list[str] | None,
    top_k: int,
    min_abs: float,
) -> list[str]:
    lines = [f"[{name}] FastKANLinear({module.in_features}->{module.out_features})"]
    grid = [float(value) for value in module.rbf.grid.detach().cpu().tolist()]
    denominator = float(module.rbf.denominator)
    spline_weight = module.spline_linear.weight.detach().cpu()
    base_weight = module.base_linear.weight.detach().cpu() if module.base_linear is not None else None
    base_bias = module.base_linear.bias.detach().cpu() if module.base_linear is not None else None
    nonzero = int(torch.count_nonzero(spline_weight).item())
    if base_weight is not None:
        nonzero += int(torch.count_nonzero(base_weight).item())
    if base_bias is not None:
        nonzero += int(torch.count_nonzero(base_bias).item())
    lines.append(f"  nonzero_terms={nonzero}, grid={grid}, h={denominator:.6g}")
    for out_idx in range(module.out_features):
        terms: list[tuple[float, str]] = []
        if base_weight is not None:
            for in_idx in range(module.in_features):
                coef = float(base_weight[out_idx, in_idx])
                terms.append((coef, f"SiLU({_source_name(source_names, in_idx)})"))
            if base_bias is not None:
                bias = float(base_bias[out_idx])
                if abs(bias) > min_abs:
                    terms.append((bias, "1"))
        for in_idx in range(module.in_features):
            for grid_idx, center in enumerate(grid):
                flat_idx = in_idx * len(grid) + grid_idx
                coef = float(spline_weight[out_idx, flat_idx])
                source = _source_name(source_names, in_idx)
                spline_source = f"LN({source})" if module.layernorm is not None else source
                terms.append((coef, f"RBF({spline_source}, center={center:.6g}, h={denominator:.6g})"))
        selected = _selected_terms(terms, top_k, min_abs)
        lines.append(f"  y{out_idx} = {_format_terms(selected)}")
    lines.append("")
    return lines


def _spline_formula_lines(
    name: str,
    module: KANLinear,
    source_names: list[str] | None,
    top_k: int,
    min_abs: float,
) -> list[str]:
    lines = [f"[{name}] KANLinear({module.in_features}->{module.out_features})"]
    base_weight = module.base_weight.detach().cpu()
    spline_weight = module.scaled_spline_weight.detach().cpu()
    knots = [float(value) for value in module.grid[0].detach().cpu().tolist()]
    nonzero = int(torch.count_nonzero(base_weight).item() + torch.count_nonzero(spline_weight).item())
    lines.append(
        f"  nonzero_terms={nonzero}, spline_order={module.spline_order}, knots={knots}"
    )
    for out_idx in range(module.out_features):
        terms: list[tuple[float, str]] = []
        for in_idx in range(module.in_features):
            source = _source_name(source_names, in_idx)
            terms.append((float(base_weight[out_idx, in_idx]), f"SiLU({source})"))
            for basis_idx in range(spline_weight.shape[-1]):
                coef = float(spline_weight[out_idx, in_idx, basis_idx])
                terms.append((coef, f"B{basis_idx}({source})"))
        selected = _selected_terms(terms, top_k, min_abs)
        lines.append(f"  y{out_idx} = {_format_terms(selected)}")
    lines.append("")
    return lines


def _linear_formula_lines(
    name: str,
    module: torch.nn.Linear,
    source_names: list[str] | None,
    top_k: int,
    min_abs: float,
) -> list[str]:
    lines = [f"[{name}] Linear({module.in_features}->{module.out_features})"]
    weight = module.weight.detach().cpu()
    bias = module.bias.detach().cpu() if module.bias is not None else None
    nonzero = int(torch.count_nonzero(weight).item())
    if bias is not None:
        nonzero += int(torch.count_nonzero(bias).item())
    lines.append(f"  nonzero_terms={nonzero}")
    for out_idx in range(module.out_features):
        terms = [
            (float(weight[out_idx, in_idx]), _source_name(source_names, in_idx))
            for in_idx in range(module.in_features)
        ]
        if bias is not None:
            terms.append((float(bias[out_idx]), "1"))
        selected = _selected_terms(terms, top_k, min_abs)
        lines.append(f"  y{out_idx} = {_format_terms(selected)}")
    lines.append("")
    return lines


def write_matbench_records(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    predictions_by_model: dict[str, dict[int, list[float]]],
    output_dir: Path,
) -> dict[str, Any]:
    if (
        args.no_matbench_records
        or args.skip_test_eval
        or args.train_size is not None
        or args.test_size is not None
    ):
        return {}

    from matbench.task import MatbenchTask
    from monty.json import MontyEncoder

    records = {}
    for model_name, fold_predictions in predictions_by_model.items():
        task = MatbenchTask(args.dataset, autoload=False)
        task.load()
        for fold, predictions in sorted(fold_predictions.items()):
            row = next(
                item
                for item in rows
                if str(item["model"]) == model_name and int(item["fold"]) == int(fold)
            )
            task.record(
                fold,
                predictions,
                params={
                    key: _json_scalar(value)
                    for key, value in row.items()
                    if not key.startswith("test_") and not key.startswith("best_val_")
                },
            )

        record_path = output_dir / f"matbench-record-{args.dataset}-{model_name}.json"
        record_path.write_text(
            json.dumps(task.as_dict(), indent=2, cls=MontyEncoder),
            encoding="utf-8",
        )
        scores = _serialize_matbench_scores(task)
        records[model_name] = {
            "record_path": str(record_path),
            "scores": scores,
        }

    return records


def _serialize_matbench_scores(task) -> Any:
    try:
        scores = task.scores
        if hasattr(scores, "as_dict"):
            return scores.as_dict()
        return json.loads(json.dumps(scores))
    except Exception as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def print_table(rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    task_type = str(rows[0].get("task_type", "regression")) if rows else "regression"
    headers = [
        "fold",
        "model",
        "featurizer_preset",
        "n_features",
        "params_before_prune",
        "params_after_prune",
        "params_pruned",
        "params_pruned_pct",
    ]
    if task_type == "classification":
        headers.extend(["best_val_rocauc", "test_rocauc", "test_accuracy", "test_f1"])
    else:
        headers.extend(["best_val_mae", "best_val_rmse", "test_mae", "test_rmse"])
    headers.extend(
        [
        "train_seconds",
        ]
    )
    widths = {
        header: max(len(header), *(len(_format(row[header])) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(_format(row[header]).ljust(widths[header]) for header in headers))


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "fold",
        "task_type",
        "model",
        "block_type",
        "kan_impl",
        "prune_kan_fraction",
        "params_before_prune",
        "params_after_prune",
        "params_pruned",
        "params_pruned_pct",
        "formula_path",
        "params",
        "effective_params",
        "pruned_params",
        "featurizer_preset",
        "n_features",
        "kan_grid_size",
        "kan_spline_order",
        "lr",
        "weight_decay",
        "loss",
        "best_epoch",
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
        "train_seconds",
        "forward_ms_per_batch",
    ]
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fold_payloads = []
    predictions_by_model: dict[str, dict[int, list[float]]] = {}
    for fold in args.folds:
        fold_data = (
            load_precomputed_feature_split(args, fold)
            if args.precomputed_feature_dir
            else load_matbench_split(args, fold)
        )
        print(
            f"\n=== {args.dataset} fold {fold}: "
            f"{fold_data['metadata']['task_type']} / "
            f"{fold_data['metadata']['input_type']} -> descriptors ===",
            flush=True,
        )
        prepared = prepare_fold(fold_data, args)
        print(
            f"Featurizer: {prepared['feature_preset']} in "
            f"{prepared['featurize_seconds']:.2f}s; selected {len(prepared['selected_features'])} features",
            flush=True,
        )
        fold_rows = []
        for model_name in args.models:
            row, predictions = run_model(model_name, fold_data, prepared, args, device)
            fold_rows.append(row)
            predictions_by_model.setdefault(str(row["model"]), {})[int(row["fold"])] = predictions
        rows.extend(fold_rows)
        fold_payloads.append(
            {
                "metadata": fold_data["metadata"],
                "feature_preset": prepared["feature_preset"],
                "featurize_seconds": prepared["featurize_seconds"],
                "selected_features": prepared["selected_features"],
                "target_scaler": prepared["target_scaler"].as_dict(),
                "train_size": prepared["train_size"],
                "val_size": prepared["val_size"],
                "test_size": prepared["test_size"],
                "results": fold_rows,
            }
        )

    print()
    print_table(rows)
    summary = summarize_rows(rows)
    matbench_records = write_matbench_records(args, rows, predictions_by_model, output_dir)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"modnet-kan-{args.dataset}-{stamp}.json"
    csv_path = output_dir / f"modnet-kan-{args.dataset}-{stamp}.csv"
    payload = {
        "args": vars(args),
        "runtime": runtime_info(device),
        "folds": fold_payloads,
        "summary": summary,
        "matbench_records": matbench_records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ordered_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
