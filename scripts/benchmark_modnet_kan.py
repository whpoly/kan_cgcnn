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
    r2_score,
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
from cgcnn_pyg_kan.pruning import (
    PruningMasks,
    apply_kan_pruning,
    iter_kan_modules,
    kan_sparsity_penalty,
)
from cgcnn_pyg_kan.spline_symbolic import (
    SPLINE_SYMBOLIC_FUNCTIONS,
    symbolify_spline_kan,
)
from cgcnn_pyg_kan.symbolic_kan import (
    SYMBOLIC_PRIMITIVES,
    SymbolicKAN,
    SymbolicRegularization,
    export_symbolic_kan,
)

FEATURE_PRESETS = [
    "auto",
    "pymatgen-composition",
    "pymatgen-structure",
    "matminer-composition",
    "matminer-structure-lite",
]
MODEL_CHOICES = [
    "mlp",
    "hybrid-fastkan",
    "hybrid-spline",
    "direct-fastkan",
    "direct-spline",
    "symbolic-kan",
    "fastkan",
    "spline",
    "kan",
]
TASK_TYPES = ("regression", "classification")
METRIC_NAMES = [
    "mae",
    "rmse",
    "r2",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "rocauc",
]
SYMBOLIC_FUNCTIONS = (
    "identity",
    "square",
    "cube",
    "sin",
    "cos",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "reciprocal",
    "product",
    "ratio",
)
DEFAULT_SYMBOLIC_FUNCTIONS = list(SYMBOLIC_FUNCTIONS)


class TargetScaler:
    def __init__(self, values: np.ndarray, mode: str = "none") -> None:
        values = np.asarray(values, dtype=np.float32)
        self.mode = mode
        if mode == "none":
            self.mean = 0.0
            self.std = 1.0
        elif mode == "standard":
            self.mean = np.asarray(values.mean(axis=0), dtype=np.float32)
            self.std = np.maximum(np.asarray(values.std(axis=0), dtype=np.float32), 1e-8)
        else:
            raise ValueError("target scaler mode must be 'none' or 'standard'")

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32, copy=False)

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        return values * std + mean

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mean": _json_ready(self.mean),
            "std": _json_ready(self.std),
        }


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MODNet-style KAN model on Matbench descriptor features."
    )
    parser.add_argument("--dataset", default="matbench_phonons")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=["hybrid-fastkan"])
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
    parser.add_argument(
        "--mlp-common-dims",
        type=int,
        nargs="+",
        default=None,
        help="MLP-only common hidden dims. Defaults to --common-dims.",
    )
    parser.add_argument(
        "--mlp-group-dims",
        type=int,
        nargs="+",
        default=None,
        help="MLP-only group hidden dims. Defaults to --group-dims.",
    )
    parser.add_argument(
        "--mlp-property-dims",
        type=int,
        nargs="+",
        default=None,
        help="MLP-only property hidden dims. Defaults to --property-dims.",
    )
    parser.add_argument(
        "--mlp-target-dims",
        type=int,
        nargs="*",
        default=None,
        help="MLP-only target hidden dims. Defaults to --target-dims.",
    )
    parser.add_argument(
        "--kan-common-dims",
        type=int,
        nargs="+",
        default=None,
        help="KAN-only common hidden dims. Defaults to --common-dims.",
    )
    parser.add_argument(
        "--kan-group-dims",
        type=int,
        nargs="+",
        default=None,
        help="KAN-only group hidden dims. Defaults to --group-dims.",
    )
    parser.add_argument(
        "--kan-property-dims",
        type=int,
        nargs="+",
        default=None,
        help="KAN-only property hidden dims. Defaults to --property-dims.",
    )
    parser.add_argument(
        "--kan-target-dims",
        type=int,
        nargs="*",
        default=None,
        help="KAN-only target hidden dims. Defaults to --target-dims.",
    )
    parser.add_argument("--kan-impl", choices=["fastkan", "spline"], default="fastkan")
    parser.add_argument("--kan-grid-size", type=int, default=5)
    parser.add_argument("--kan-spline-order", type=int, default=3)
    parser.set_defaults(kan_layernorm=True)
    parser.add_argument(
        "--kan-layernorm",
        dest="kan_layernorm",
        action="store_true",
        help="Use LayerNorm between hidden KAN layers (default).",
    )
    parser.add_argument(
        "--no-kan-layernorm",
        dest="kan_layernorm",
        action="store_false",
        help=(
            "Keep hidden KAN layers as pure sums of edge functions. Required "
            "for native compositional KAN symbolification."
        ),
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "elu", "silu"],
        default="elu",
        help="MLP trunk activation; ELU matches the official MODNet Matbench fit settings.",
    )
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
            "Prune this fraction of KAN edges or coefficients after training. "
            "The MLP trunk is never pruned."
        ),
    )
    parser.add_argument(
        "--prune-mode",
        choices=["edge", "parameter"],
        default="edge",
        help="edge performs interpretable structured KAN-edge pruning; parameter is scalar ablation.",
    )
    parser.add_argument(
        "--prune-finetune-epochs",
        type=int,
        default=0,
        help="Fine-tune after pruning while enforcing the pruning masks.",
    )
    parser.add_argument(
        "--kan-l1-lambda",
        type=float,
        default=0.0,
        help=(
            "Strength of --kan-sparsity-mode. Keep at 0 for the official accuracy "
            "benchmark; use a positive fixed value for the separate sparse model."
        ),
    )
    parser.add_argument(
        "--kan-sparsity-mode",
        choices=["edge-group", "parameter-l1"],
        default="edge-group",
        help=(
            "edge-group penalises the same input-output edge groups later removed by "
            "structured pruning; parameter-l1 is retained as an ablation."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--inner-fold-index",
        type=int,
        default=None,
        help=(
            "Use this deterministic inner CV fold inside the current Matbench outer "
            "train+validation partition. Intended for strict nested tuning."
        ),
    )
    parser.add_argument("--inner-n-splits", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument(
        "--early-stopping-monitor",
        choices=["validation", "loss", "none"],
        default="validation",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.set_defaults(restore_best_state=True)
    parser.add_argument(
        "--restore-best-state",
        dest="restore_best_state",
        action="store_true",
    )
    parser.add_argument(
        "--no-restore-best-state",
        dest="restore_best_state",
        action="store_false",
    )
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
        help=(
            "Write readable explicit formula files for KAN-family models after "
            "optional pruning. Use --formula-top-k 0 for exact untruncated formulas."
        ),
    )
    parser.add_argument(
        "--formula-top-k",
        type=int,
        default=40,
        help=(
            "Maximum nonzero terms per neuron in exported formula files. "
            "Use 0 to write every nonzero term after pruning."
        ),
    )
    parser.add_argument(
        "--formula-min-abs",
        type=float,
        default=0.0,
        help="Minimum absolute coefficient to include in exported formula files.",
    )
    parser.add_argument(
        "--distill-simple-formula",
        action="store_true",
        help=(
            "Distill a sparse symbolic surrogate from the trained model using at "
            "most --simple-formula-max-inputs descriptors. This is a post-hoc "
            "interpretability experiment, not a tuning metric."
        ),
    )
    parser.add_argument("--simple-formula-min-inputs", type=int, default=5)
    parser.add_argument("--simple-formula-max-inputs", type=int, default=10)
    parser.add_argument("--simple-formula-max-terms", type=int, default=10)
    parser.add_argument(
        "--simple-formula-method",
        choices=["symbolic", "polynomial"],
        default="symbolic",
        help="Use a protected common-function library or the legacy degree-1/2 polynomial library.",
    )
    parser.add_argument("--simple-formula-degree", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--simple-formula-functions",
        nargs="+",
        choices=SYMBOLIC_FUNCTIONS,
        default=DEFAULT_SYMBOLIC_FUNCTIONS,
        help=(
            "Candidate functions for symbolic regression. log/sqrt use abs(x), "
            "reciprocal/ratio use a protected denominator, and exp is clipped."
        ),
    )
    parser.add_argument("--simple-formula-epsilon", type=float, default=1e-3)
    parser.add_argument("--simple-formula-exp-clip", type=float, default=8.0)
    parser.add_argument(
        "--simple-formula-coverage",
        type=float,
        default=0.95,
        help=(
            "Requested split-conformal marginal coverage for the formula-to-model "
            "absolute error band."
        ),
    )
    parser.add_argument(
        "--simple-formula-calibration-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of the outer train+validation partition reserved before model "
            "training for formula fidelity calibration."
        ),
    )
    parser.add_argument("--symbolic-hidden-dims", type=int, nargs="+", default=[4])
    parser.add_argument("--symbolic-edges-per-unit", type=int, default=3)
    parser.add_argument(
        "--symbolic-primitives",
        nargs="+",
        choices=SYMBOLIC_PRIMITIVES,
        default=list(SYMBOLIC_PRIMITIVES),
    )
    parser.add_argument("--symbolic-temperature-start", type=float, default=2.0)
    parser.add_argument("--symbolic-temperature-end", type=float, default=0.1)
    parser.add_argument("--symbolic-selection-lambda", type=float, default=1e-3)
    parser.add_argument("--symbolic-entropy-weight", type=float, default=1.0)
    parser.add_argument("--symbolic-nms-weight", type=float, default=0.1)
    parser.add_argument("--symbolic-unit-weight", type=float, default=1e-3)
    parser.add_argument("--symbolic-bias-weight", type=float, default=1e-4)
    parser.add_argument("--symbolic-projection-l1", type=float, default=1e-5)
    parser.add_argument("--symbolic-target-density", type=float, default=0.75)
    parser.add_argument("--symbolic-gate-lr-scale", type=float, default=0.2)
    parser.add_argument("--symbolic-unit-threshold", type=float, default=0.5)
    parser.add_argument("--symbolic-projection-top-k", type=int, default=3)
    parser.add_argument("--symbolic-hardening-epochs", type=int, default=100)
    parser.add_argument("--symbolic-hardening-lr", type=float, default=1e-4)
    parser.add_argument(
        "--symbolify-spline-kan",
        action="store_true",
        help=(
            "Apply KAN-paper/pykan-style edge auto-symbolification to a trained "
            "two-layer direct spline KAN and evaluate the replaced formula."
        ),
    )
    parser.add_argument(
        "--spline-symbolic-functions",
        nargs="+",
        choices=SPLINE_SYMBOLIC_FUNCTIONS,
        default=list(SPLINE_SYMBOLIC_FUNCTIONS),
    )
    parser.add_argument("--spline-symbolic-input-edges", type=int, default=5)
    parser.add_argument("--spline-symbolic-output-edges", type=int, default=4)
    parser.add_argument("--spline-symbolic-max-fit-samples", type=int, default=1024)
    parser.add_argument("--spline-symbolic-search-range", type=float, default=10.0)
    parser.add_argument("--spline-symbolic-grid-size", type=int, default=21)
    parser.add_argument("--spline-symbolic-iterations", type=int, default=2)
    parser.add_argument("--spline-symbolic-complexity-weight", type=float, default=0.2)
    parser.add_argument("--spline-symbolic-epsilon", type=float, default=1e-3)
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


def is_kan_family(family: str) -> bool:
    return family != "mlp"


def kan_impl_for_family(family: str, fallback: str = "fastkan") -> str:
    if family.endswith("spline") or family == "spline":
        return "spline"
    if family.endswith("fastkan") or family == "fastkan":
        return "fastkan"
    return fallback


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


def make_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    if isinstance(model, SymbolicKAN):
        groups = [
            {
                "params": model.continuous_parameters(),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": model.gate_parameters(),
                "lr": args.lr * args.symbolic_gate_lr_scale,
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(groups) if args.weight_decay else torch.optim.Adam(groups)
    if args.weight_decay == 0:
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    return torch.optim.AdamW(
        adamw_parameter_groups(model, args.weight_decay),
        lr=args.lr,
    )


def select_subset(items, targets, size: int | None, seed: int):
    targets_arr = np.asarray(targets, dtype=np.float32)
    if size is None or size >= len(items):
        return list(items), targets_arr
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(items), size=size, replace=False)
    selected_items = [
        items.iloc[int(i)] if hasattr(items, "iloc") else items[int(i)]
        for i in indices
    ]
    return selected_items, targets_arr[indices]


def as_2d_targets(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim == 2:
        return values
    raise ValueError(f"targets must be 1D or 2D, got shape {values.shape}")


def target_names_from_metadata(metadata: dict[str, Any]) -> list[str]:
    names = metadata.get("target_names")
    if names:
        return [str(name) for name in names]
    return [str(metadata.get("target", "target"))]


def target_display_name(names: list[str]) -> str:
    return names[0] if len(names) == 1 else "+".join(names)


def _target_metric_suffix(name: str) -> str:
    return "__" + _safe_symbol(name)


def feature_selection_target(values: np.ndarray) -> np.ndarray:
    values = as_2d_targets(values)
    if values.shape[1] == 1:
        return values[:, 0]
    means = values.mean(axis=0, keepdims=True)
    stds = np.maximum(values.std(axis=0, keepdims=True), 1e-8)
    return ((values - means) / stds).mean(axis=1)


def load_matbench_split(args: argparse.Namespace, fold: int) -> dict[str, Any]:
    from matbench.metadata import mbv01_metadata
    from matbench.task import MatbenchTask

    if args.dataset == "matbench_elastic":
        return load_elastic_matbench_split(args, fold)

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
            "target_names": [metadata.target],
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


def load_elastic_matbench_split(args: argparse.Namespace, fold: int) -> dict[str, Any]:
    from matbench.metadata import mbv01_metadata
    from matbench.task import MatbenchTask

    args.loss = validate_task_loss("regression", args.loss)
    source_tasks = ["matbench_log_gvrh", "matbench_log_kvrh"]
    loaded_tasks = []
    train_inputs = None
    test_inputs = None
    train_targets = []
    test_targets = []
    official_test_size = 0
    for task_name in source_tasks:
        task = MatbenchTask(task_name, autoload=False)
        task.load()
        fold_train_inputs, fold_train_targets = task.get_train_and_val_data(fold)
        if train_inputs is None:
            train_inputs = fold_train_inputs
        else:
            assert_aligned_materials(train_inputs, fold_train_inputs, task_name)
        train_targets.append(np.asarray(fold_train_targets, dtype=np.float32))

        if args.skip_test_eval:
            fold_key = task.folds_map[fold]
            official_test_size = len(task.validation[fold_key].test)
        else:
            fold_test_inputs, fold_test_targets = task.get_test_data(fold, include_target=True)
            if test_inputs is None:
                test_inputs = fold_test_inputs
            else:
                assert_aligned_materials(test_inputs, fold_test_inputs, task_name)
            test_targets.append(np.asarray(fold_test_targets, dtype=np.float32))
            official_test_size = len(fold_test_targets)
        loaded_tasks.append(task_name)

    if train_inputs is None:
        raise RuntimeError("No elastic train inputs were loaded")
    target_names = [str(mbv01_metadata[task].target) for task in source_tasks]
    train_y = np.column_stack(train_targets).astype(np.float32)
    train_inputs, train_y = select_subset(train_inputs, train_y, args.train_size, args.seed)
    if args.skip_test_eval:
        test_y = np.empty((0, len(target_names)), dtype=np.float32)
        test_size = 0
    else:
        if test_inputs is None:
            raise RuntimeError("No elastic test inputs were loaded")
        test_y = np.column_stack(test_targets).astype(np.float32)
        test_inputs, test_y = select_subset(test_inputs, test_y, args.test_size, args.seed + 1)
        test_size = len(test_inputs)

    return {
        "metadata": {
            "dataset": args.dataset,
            "fold": fold,
            "task_type": "regression",
            "target": target_display_name(target_names),
            "target_names": target_names,
            "source_tasks": loaded_tasks,
            "input_type": "structure",
            "unit": None,
            "mad": None,
            "frac_true": None,
            "n_samples": int(mbv01_metadata["matbench_log_gvrh"].n_samples),
            "train_and_val_size": len(train_inputs),
            "test_size": test_size,
            "official_test_size": official_test_size,
        },
        "train_inputs": train_inputs,
        "train_targets": train_y,
        "test_inputs": test_inputs,
        "test_targets": test_y,
    }


def assert_aligned_materials(reference: Any, candidate: Any, task_name: str) -> None:
    if len(reference) != len(candidate):
        raise RuntimeError(f"{task_name} split length does not match the elastic reference")
    for idx, (left, right) in enumerate(zip(reference, candidate)):
        left_formula = getattr(getattr(left, "composition", None), "reduced_formula", str(left))
        right_formula = getattr(getattr(right, "composition", None), "reduced_formula", str(right))
        if left_formula != right_formula:
            raise RuntimeError(
                f"{task_name} material order differs at row {idx}: {left_formula} != {right_formula}"
            )


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
    synthetic_elastic = args.dataset == "matbench_elastic"
    metadata = None if synthetic_elastic else mbv01_metadata[args.dataset]
    task_type = "regression" if synthetic_elastic else str(metadata.task_type)
    input_type = "structure" if synthetic_elastic else str(metadata.input_type)
    if task_type not in TASK_TYPES:
        raise ValueError(
            f"{args.dataset} task_type is {task_type!r}; expected one of {TASK_TYPES}"
        )
    args.loss = validate_task_loss(task_type, args.loss)
    if task_type == "classification" and args.target_scale != "none":
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
    if synthetic_elastic:
        target_columns = list(train_targets_frame.columns)
        if len(target_columns) < 2:
            raise ValueError(
                "matbench_elastic precomputed target files must contain both elastic targets"
            )
        missing = [column for column in target_columns if column not in test_targets_frame.columns]
        if missing:
            raise ValueError(f"test target file is missing columns: {missing}")
    else:
        target_column = str(metadata.target)
        if target_column not in train_targets_frame.columns:
            target_column = str(train_targets_frame.columns[0])
        if target_column not in test_targets_frame.columns:
            target_column = str(test_targets_frame.columns[0])
        target_columns = [target_column]

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
            "task_type": task_type,
            "target": target_display_name([str(column) for column in target_columns]),
            "target_names": [str(column) for column in target_columns],
            "source_tasks": ["matbench_log_gvrh", "matbench_log_kvrh"] if synthetic_elastic else [args.dataset],
            "input_type": input_type,
            "unit": None if synthetic_elastic else getattr(metadata, "unit", None),
            "mad": None if synthetic_elastic else getattr(metadata, "mad", None),
            "frac_true": None if synthetic_elastic else getattr(metadata, "frac_true", None),
            "n_samples": len(train_features) + len(test_features) if synthetic_elastic else metadata.n_samples,
            "train_and_val_size": len(train_features),
            "test_size": len(test_features),
            "official_test_size": len(test_features),
            "precomputed_feature_dir": str(fold_dir),
        },
        "precomputed_features": True,
        "train_features": train_features,
        "test_features": test_features,
        "train_targets": train_targets_frame[target_columns].to_numpy(dtype=np.float32),
        "test_targets": test_targets_frame[target_columns].to_numpy(dtype=np.float32),
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
    inner_fold_index: int | None = None,
    inner_n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    if inner_fold_index is not None:
        if inner_n_splits < 2:
            raise ValueError("inner_n_splits must be at least 2")
        if not 0 <= inner_fold_index < inner_n_splits:
            raise ValueError(
                f"inner_fold_index must be in [0, {inner_n_splits}), got {inner_fold_index}"
            )
        if size < inner_n_splits:
            raise ValueError(
                f"Cannot make {inner_n_splits} inner folds from only {size} samples"
            )
        if task_type == "classification" and targets is not None:
            labels = np.asarray(targets).astype(int).reshape(-1)
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) >= 2 and int(counts.min()) >= inner_n_splits:
                from sklearn.model_selection import StratifiedKFold

                splitter = StratifiedKFold(
                    n_splits=inner_n_splits,
                    shuffle=True,
                    random_state=seed,
                )
                return list(splitter.split(np.zeros(size), labels))[inner_fold_index]
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=inner_n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.arange(size)))[inner_fold_index]
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


def reserve_formula_calibration(
    train_indices: np.ndarray,
    targets: np.ndarray,
    args: argparse.Namespace,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not args.distill_simple_formula:
        return train_indices, np.array([], dtype=int)
    ratio = float(args.simple_formula_calibration_ratio)
    if not 0.0 < ratio < 0.5:
        raise ValueError("--simple-formula-calibration-ratio must be in (0, 0.5)")
    if args.inner_fold_index is not None or args.val_ratio != 0.0:
        raise ValueError(
            "Formula calibration must be reserved from a fixed-epoch final fit; "
            "use --val-ratio 0 and omit --inner-fold-index"
        )
    local_train, local_calibration = split_train_val(
        len(train_indices),
        ratio,
        args.seed + 104729,
        targets=np.asarray(targets)[train_indices],
        task_type=task_type,
    )
    return train_indices[local_train], train_indices[local_calibration]


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
    target_names: list[str],
    args: argparse.Namespace,
) -> torch.nn.Module:
    family = canonical_model_name(model_name, args)
    if family == "symbolic-kan":
        return SymbolicKAN(
            in_features=n_feat,
            target_names=target_names,
            hidden_dims=clean_hidden_dims(args.symbolic_hidden_dims),
            edges_per_unit=args.symbolic_edges_per_unit,
            primitives=args.symbolic_primitives,
            temperature_start=args.symbolic_temperature_start,
            temperature_end=args.symbolic_temperature_end,
            regularization=SymbolicRegularization(
                selection=args.symbolic_selection_lambda,
                entropy=args.symbolic_entropy_weight,
                nms=args.symbolic_nms_weight,
                unit=args.symbolic_unit_weight,
                bias=args.symbolic_bias_weight,
                projection_l1=args.symbolic_projection_l1,
                target_density=args.symbolic_target_density,
            ),
        )
    common_dims, group_dims, property_dims, target_dims = model_dims_for_family(family, args)
    if family.startswith("hybrid-"):
        block_types = ("mlp", "mlp", "mlp", "kan")
        output_head_type = "kan"
        architecture = "modnet-mlp-trunk-kan-head"
    elif family.startswith("direct-"):
        block_types = ("mlp", "mlp", "mlp", "kan")
        output_head_type = "kan"
        architecture = "descriptor-kan"
    elif family == "mlp":
        block_types = ("mlp", "mlp", "mlp", "mlp")
        output_head_type = "linear"
        architecture = "modnet-mlp"
    else:
        block_types = ("kan", "kan", "kan", "kan")
        output_head_type = "kan"
        architecture = "compact-all-kan"
    model = MODNetKAN(
        n_feat=n_feat,
        targets=[[target_names]],
        num_neurons=(
            common_dims,
            group_dims,
            property_dims,
            target_dims,
        ),
        block_type="mlp" if family == "mlp" else "kan",
        block_types=block_types,
        output_head_type=output_head_type,
        mlp_activation=args.activation,
        kan_impl=kan_impl_for_family(family, args.kan_impl),
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
        kan_use_layernorm=args.kan_layernorm,
        dropout=args.dropout,
    )
    model.architecture = architecture  # type: ignore[attr-defined]
    return model


def canonical_model_name(model_name: str, args: argparse.Namespace) -> str:
    if model_name == "kan":
        return args.kan_impl
    return model_name


def model_dims_for_family(
    family: str,
    args: argparse.Namespace,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if family == "symbolic-kan":
        return ([], [], [], clean_hidden_dims(args.symbolic_hidden_dims))
    if family == "mlp":
        return (
            clean_hidden_dims(args.mlp_common_dims or args.common_dims),
            clean_hidden_dims(args.mlp_group_dims or args.group_dims),
            clean_hidden_dims(args.mlp_property_dims or args.property_dims),
            clean_hidden_dims(args.mlp_target_dims if args.mlp_target_dims is not None else args.target_dims),
        )
    if family.startswith("direct-"):
        return (
            [],
            [],
            [],
            clean_hidden_dims(
                args.kan_target_dims if args.kan_target_dims is not None else args.target_dims
            ),
        )
    if family.startswith("hybrid-"):
        return (
            clean_hidden_dims(args.mlp_common_dims or args.common_dims),
            clean_hidden_dims(args.mlp_group_dims or args.group_dims),
            clean_hidden_dims(args.mlp_property_dims or args.property_dims),
            clean_hidden_dims(
                args.kan_target_dims if args.kan_target_dims is not None else args.target_dims
            ),
        )
    return (
        clean_hidden_dims(args.kan_common_dims or args.common_dims),
        clean_hidden_dims(args.kan_group_dims or args.group_dims),
        clean_hidden_dims(args.kan_property_dims or args.property_dims),
        clean_hidden_dims(args.kan_target_dims if args.kan_target_dims is not None else args.target_dims),
    )


def clean_hidden_dims(values: list[int] | tuple[int, ...]) -> list[int]:
    return [int(value) for value in values if int(value) > 0]


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
    kan_l1_lambda: float = 0.0,
    kan_sparsity_mode: str = "edge-group",
    pruning_masks: PruningMasks | None = None,
    use_symbolic_regularization: bool = True,
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
        if kan_l1_lambda > 0:
            loss = loss + kan_l1_lambda * kan_sparsity_penalty(
                model, kan_sparsity_mode
            )
        if use_symbolic_regularization and isinstance(model, SymbolicKAN):
            loss = loss + model.symbolic_regularization()
        loss.backward()
        optimizer.step()
        if pruning_masks is not None:
            pruning_masks.enforce()
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
    target_names: list[str] | None = None,
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

    pred_2d = pred.reshape(pred.shape[0], -1)
    target_2d = target.reshape(target.shape[0], -1)
    mae = torch.mean(torch.abs(pred_2d - target_2d)).item()
    rmse = torch.sqrt(F.mse_loss(pred_2d, target_2d)).item()
    metrics = {name: float("nan") for name in METRIC_NAMES}
    metrics["mae"] = mae
    metrics["rmse"] = rmse
    if len(pred_2d) >= 2:
        try:
            metrics["r2"] = float(
                r2_score(
                    target_2d.numpy().reshape(-1),
                    pred_2d.numpy().reshape(-1),
                )
            )
        except ValueError:
            metrics["r2"] = float("nan")
    names = target_names or [f"target_{idx}" for idx in range(pred_2d.shape[1])]
    for idx, name in enumerate(names[: pred_2d.shape[1]]):
        suffix = _target_metric_suffix(name)
        pred_col = pred_2d[:, idx]
        target_col = target_2d[:, idx]
        metrics[f"mae{suffix}"] = torch.mean(torch.abs(pred_col - target_col)).item()
        metrics[f"rmse{suffix}"] = torch.sqrt(F.mse_loss(pred_col, target_col)).item()
        if len(pred_col) >= 2:
            try:
                metrics[f"r2{suffix}"] = float(
                    r2_score(target_col.numpy().reshape(-1), pred_col.numpy().reshape(-1))
                )
            except ValueError:
                metrics[f"r2{suffix}"] = float("nan")
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
    target_names: list[str] | None = None,
) -> dict[str, float]:
    pred, target = predict_loader(model, loader, scaler, device, task_type)
    return metrics_from_predictions(pred, target, task_type, target_names=target_names)


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


def feature_preprocessing_affine(
    prepared: dict[str, Any],
) -> tuple[list[float], list[float], list[float]]:
    pipeline = prepared.get("feature_pipeline")
    if pipeline is None:
        processor = prepared.get("processor")
        pipeline = getattr(processor, "pipeline_", None)
    if pipeline is None:
        raise ValueError(
            "Symbolic-KAN raw formula export requires a fitted feature pipeline"
        )

    n_features = len(prepared["selected_features"])
    imputer = pipeline.named_steps.get("imputer")
    if imputer is None or not hasattr(imputer, "statistics_"):
        raise ValueError(
            "Symbolic-KAN raw formula export requires fitted imputation statistics"
        )
    impute_values = np.asarray(imputer.statistics_, dtype=float)

    scaler = pipeline.named_steps.get("scaler")
    if isinstance(scaler, MinMaxScaler):
        scales = np.asarray(scaler.scale_, dtype=float)
        offsets = np.asarray(scaler.min_, dtype=float)
    elif isinstance(scaler, StandardScaler):
        standard_scale = np.asarray(scaler.scale_, dtype=float)
        scales = 1.0 / standard_scale
        offsets = -np.asarray(scaler.mean_, dtype=float) / standard_scale
    elif scaler is None:
        scales = np.ones(n_features, dtype=float)
        offsets = np.zeros(n_features, dtype=float)
    else:
        raise TypeError(
            "Unsupported feature scaler for Symbolic-KAN raw formula export: "
            f"{type(scaler).__name__}"
        )

    if (
        len(scales) != n_features
        or len(offsets) != n_features
        or len(impute_values) != n_features
    ):
        raise ValueError(
            "Fitted feature preprocessing parameters do not match selected features"
        )
    return scales.tolist(), offsets.tolist(), impute_values.tolist()


def harden_and_evaluate_symbolic_kan(
    model: SymbolicKAN,
    train_loader: DataLoader,
    test_loader: DataLoader,
    prepared: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    task_type: str,
    target_names: list[str],
    soft_prediction: torch.Tensor,
    test_target: torch.Tensor,
    fold: int,
) -> dict[str, float | int | str]:
    model.harden(
        unit_threshold=args.symbolic_unit_threshold,
        projection_top_k=args.symbolic_projection_top_k,
    )
    hard_optimizer = torch.optim.Adam(
        model.continuous_parameters(),
        lr=args.symbolic_hardening_lr,
    )
    start = time.perf_counter()
    for epoch in range(1, args.symbolic_hardening_epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            hard_optimizer,
            device,
            task_type,
            args.loss,
            use_symbolic_regularization=False,
        )
        if args.log_every_epochs > 0 and (
            epoch == 1
            or epoch == args.symbolic_hardening_epochs
            or epoch % args.log_every_epochs == 0
        ):
            print(
                f"[symbolic-kan] hardening fine-tune "
                f"{epoch}/{args.symbolic_hardening_epochs} loss={loss:.6g}",
                flush=True,
            )
    hardening_seconds = time.perf_counter() - start

    hard_prediction, hard_target = predict_loader(
        model,
        test_loader,
        prepared["target_scaler"],
        device,
        task_type,
    )
    hard_metrics = metrics_from_predictions(
        hard_prediction,
        hard_target,
        task_type,
        target_names=target_names,
    )
    soft_2d = soft_prediction.reshape(len(soft_prediction), -1)
    hard_2d = hard_prediction.reshape(len(hard_prediction), -1)
    target_2d = test_target.reshape(len(test_target), -1)
    fidelity_r2_values = []

    scaler_mean = np.broadcast_to(
        np.asarray(prepared["target_scaler"].mean, dtype=float),
        (len(target_names),),
    )
    scaler_std = np.broadcast_to(
        np.asarray(prepared["target_scaler"].std, dtype=float),
        (len(target_names),),
    )
    feature_scales, feature_offsets, feature_impute_values = (
        feature_preprocessing_affine(prepared)
    )
    payload, text = export_symbolic_kan(
        model,
        prepared["selected_features"],
        target_means=scaler_mean.tolist(),
        target_stds=scaler_std.tolist(),
        feature_scales=feature_scales,
        feature_offsets=feature_offsets,
        feature_impute_values=feature_impute_values,
    )
    for target_index, record in enumerate(payload["targets"]):
        hard_column = hard_2d[:, target_index]
        soft_column = soft_2d[:, target_index]
        target_column = target_2d[:, target_index]
        fidelity_r2 = (
            float(
                r2_score(
                    soft_column.numpy(),
                    hard_column.numpy(),
                )
            )
            if len(hard_column) >= 2
            else float("nan")
        )
        fidelity_r2_values.append(fidelity_r2)
        record["test_target_mae"] = float(
            torch.mean(torch.abs(hard_column - target_column))
        )
        record["test_teacher_mae"] = float(
            torch.mean(torch.abs(hard_column - soft_column))
        )
        record["test_fidelity_r2"] = fidelity_r2

    output_dir = Path(args.output_dir)
    text_path = output_dir / (
        f"symbolic-kan-formula-{args.dataset}-fold{fold}.txt"
    )
    json_path = output_dir / (
        f"symbolic-kan-formula-{args.dataset}-fold{fold}.json"
    )
    payload.update(
        {
            "fold": fold,
            "soft_test_mae": float(
                torch.mean(torch.abs(soft_2d - target_2d))
            ),
            "hard_test_mae": hard_metrics["mae"],
            "hardening_epochs": args.symbolic_hardening_epochs,
            "hardening_seconds": hardening_seconds,
            "unit_threshold": args.symbolic_unit_threshold,
            "projection_top_k": args.symbolic_projection_top_k,
        }
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metric_lines = [
        "",
        f"soft_test_mae = {payload['soft_test_mae']:.8g}",
        f"hard_formula_test_mae = {hard_metrics['mae']:.8g}",
        "hard_formula_to_soft_r2 = "
        + f"{float(np.mean(fidelity_r2_values)):.8g}",
    ]
    text_path.write_text(text + "\n".join(metric_lines), encoding="utf-8")

    active_features = sorted(
        {
            feature
            for record in payload["targets"]
            for feature in record["active_feature_names"]
        }
    )
    operators = sorted(
        {
            operator
            for record in payload["targets"]
            for operator in record["operators"]
        }
    )
    active_units = sum(
        int(record["n_active_units"]) for record in payload["targets"]
    )
    return {
        "symbolic_kan_formula_path": str(text_path),
        "symbolic_kan_json_path": str(json_path),
        "symbolic_kan_soft_test_mae": float(
            torch.mean(torch.abs(soft_2d - target_2d))
        ),
        "symbolic_kan_hard_test_mae": hard_metrics["mae"],
        "symbolic_kan_hard_test_rmse": hard_metrics["rmse"],
        "symbolic_kan_hard_test_r2": hard_metrics["r2"],
        "symbolic_kan_test_teacher_mae": float(
            torch.mean(torch.abs(hard_2d - soft_2d))
        ),
        "symbolic_kan_test_fidelity_r2_pct": 100.0
        * float(np.mean(fidelity_r2_values)),
        "symbolic_kan_active_features": ",".join(active_features),
        "symbolic_kan_operators": ",".join(operators),
        "symbolic_kan_active_units": active_units,
        "symbolic_kan_hardening_epochs": args.symbolic_hardening_epochs,
        "symbolic_kan_hardening_seconds": hardening_seconds,
    }


def symbolify_and_evaluate_spline_kan(
    model: torch.nn.Module,
    prepared: dict[str, Any],
    args: argparse.Namespace,
    target_names: list[str],
    teacher_prediction: torch.Tensor,
    test_target: torch.Tensor,
    fold: int,
) -> dict[str, float | int | str]:
    layers = [
        module
        for module in iter_kan_modules(model)
        if isinstance(module, KANLinear)
    ]
    symbolic_scaled, payload, text = symbolify_spline_kan(
        layers,
        prepared["x_formula_train"],
        prepared["x_test"],
        prepared["selected_features"],
        target_names,
        functions=args.spline_symbolic_functions,
        input_edges_per_hidden=args.spline_symbolic_input_edges,
        output_edges_per_target=args.spline_symbolic_output_edges,
        max_fit_samples=args.spline_symbolic_max_fit_samples,
        search_range=args.spline_symbolic_search_range,
        grid_size=args.spline_symbolic_grid_size,
        iterations=args.spline_symbolic_iterations,
        complexity_weight=args.spline_symbolic_complexity_weight,
        epsilon=args.spline_symbolic_epsilon,
    )
    symbolic_prediction = prepared["target_scaler"].inverse_transform_tensor(
        torch.from_numpy(symbolic_scaled)
    )
    teacher_2d = teacher_prediction.reshape(len(teacher_prediction), -1)
    formula_2d = symbolic_prediction.reshape(len(symbolic_prediction), -1)
    target_2d = test_target.reshape(len(test_target), -1)
    formula_metrics = metrics_from_predictions(
        symbolic_prediction,
        test_target,
        "regression",
        target_names=target_names,
    )
    fidelity_values = []
    for target_index, record in enumerate(payload["targets"]):
        formula_column = formula_2d[:, target_index]
        teacher_column = teacher_2d[:, target_index]
        target_column = target_2d[:, target_index]
        fidelity = (
            float(r2_score(teacher_column.numpy(), formula_column.numpy()))
            if len(formula_column) >= 2
            else float("nan")
        )
        fidelity_values.append(fidelity)
        record["test_target_mae"] = float(
            torch.mean(torch.abs(formula_column - target_column))
        )
        record["test_teacher_mae"] = float(
            torch.mean(torch.abs(formula_column - teacher_column))
        )
        record["test_fidelity_r2"] = fidelity

    output_dir = Path(args.output_dir)
    text_path = output_dir / (
        f"spline-symbolic-formula-{args.dataset}-fold{fold}.txt"
    )
    json_path = output_dir / (
        f"spline-symbolic-formula-{args.dataset}-fold{fold}.json"
    )
    payload.update(
        {
            "fold": fold,
            "teacher_test_mae": float(
                torch.mean(torch.abs(teacher_2d - target_2d))
            ),
            "formula_test_mae": formula_metrics["mae"],
        }
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text_path.write_text(
        text
        + "\n"
        + f"teacher_test_mae = {payload['teacher_test_mae']:.8g}\n"
        + f"symbolic_formula_test_mae = {formula_metrics['mae']:.8g}\n"
        + "formula_to_teacher_r2 = "
        + f"{float(np.mean(fidelity_values)):.8g}\n",
        encoding="utf-8",
    )
    active_features = sorted(
        {
            feature
            for record in payload["targets"]
            for feature in record["active_feature_names"]
        }
    )
    operators = sorted(
        {
            operator
            for record in payload["targets"]
            for operator in record["operators"]
        }
    )
    edge_count = sum(int(record["n_edges"]) for record in payload["targets"])
    return {
        "spline_symbolic_formula_path": str(text_path),
        "spline_symbolic_json_path": str(json_path),
        "spline_symbolic_test_target_mae": formula_metrics["mae"],
        "spline_symbolic_test_teacher_mae": float(
            torch.mean(torch.abs(formula_2d - teacher_2d))
        ),
        "spline_symbolic_test_fidelity_r2_pct": 100.0
        * float(np.mean(fidelity_values)),
        "spline_symbolic_active_features": ",".join(active_features),
        "spline_symbolic_operators": ",".join(operators),
        "spline_symbolic_n_edges": edge_count,
    }


@torch.no_grad()
def predict_features(
    model: torch.nn.Module,
    features: np.ndarray,
    scaler: TargetScaler,
    device: torch.device,
    task_type: str,
    batch_size: int,
) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0, 1), dtype=np.float32)
    model.eval()
    predictions = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        raw = model(batch)
        raw = raw.reshape(len(batch), -1)
        if task_type == "classification":
            prediction = torch.sigmoid(raw)
        else:
            prediction = scaler.inverse_transform_tensor(raw)
        predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _polynomial_library(
    values: np.ndarray,
    variable_names: list[str],
    degree: int,
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for idx, name in enumerate(variable_names):
        columns.append(values[:, idx])
        names.append(name)
    if degree >= 2:
        for left in range(values.shape[1]):
            for right in range(left, values.shape[1]):
                columns.append(values[:, left] * values[:, right])
                if left == right:
                    names.append(f"{variable_names[left]}^2")
                else:
                    names.append(f"{variable_names[left]}*{variable_names[right]}")
    matrix = np.column_stack(columns) if columns else np.empty((len(values), 0))
    return matrix.astype(float, copy=False), names


def _protected_denominator(values: np.ndarray, epsilon: float) -> np.ndarray:
    signs = np.where(values < 0.0, -1.0, 1.0)
    return signs * np.maximum(np.abs(values), epsilon)


def _symbolic_library(
    values: np.ndarray,
    variable_names: list[str],
    functions: list[str] | tuple[str, ...],
    epsilon: float,
    exp_clip: float,
) -> tuple[np.ndarray, list[str]]:
    """Build a finite, human-readable library of protected symbolic terms."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(variable_names):
        raise ValueError("Symbolic values and variable names have incompatible shapes")
    if epsilon <= 0 or exp_clip <= 0:
        raise ValueError("Symbolic epsilon and exp clip must be positive")
    unknown = sorted(set(functions) - set(SYMBOLIC_FUNCTIONS))
    if unknown:
        raise ValueError(f"Unsupported symbolic functions: {unknown}")
    enabled = list(dict.fromkeys(functions))
    columns: list[np.ndarray] = []
    names: list[str] = []

    def append(column: np.ndarray, name: str) -> None:
        column = np.nan_to_num(
            np.asarray(column, dtype=float),
            nan=0.0,
            posinf=np.finfo(float).max ** 0.25,
            neginf=-(np.finfo(float).max ** 0.25),
        )
        columns.append(column)
        names.append(name)

    for idx, name in enumerate(variable_names):
        value = values[:, idx]
        if "identity" in enabled:
            append(value, name)
        if "square" in enabled:
            append(value**2, f"({name})^2")
        if "cube" in enabled:
            append(value**3, f"({name})^3")
        if "sin" in enabled:
            append(np.sin(value), f"sin({name})")
        if "cos" in enabled:
            append(np.cos(value), f"cos({name})")
        if "tanh" in enabled:
            append(np.tanh(value), f"tanh({name})")
        if "exp" in enabled:
            append(
                np.exp(np.clip(value, -exp_clip, exp_clip)),
                f"exp(clip({name}, {-exp_clip:.6g}, {exp_clip:.6g}))",
            )
        if "log" in enabled:
            append(np.log(np.abs(value) + epsilon), f"log(abs({name})+{epsilon:.6g})")
        if "sqrt" in enabled:
            append(np.sqrt(np.abs(value)), f"sqrt(abs({name}))")
        if "reciprocal" in enabled:
            append(
                1.0 / _protected_denominator(value, epsilon),
                f"1/protected({name}, eps={epsilon:.6g})",
            )

    if "product" in enabled:
        for left in range(values.shape[1]):
            for right in range(left + 1, values.shape[1]):
                append(
                    values[:, left] * values[:, right],
                    f"({variable_names[left]})*({variable_names[right]})",
                )
    if "ratio" in enabled:
        for numerator in range(values.shape[1]):
            for denominator in range(values.shape[1]):
                if numerator == denominator:
                    continue
                append(
                    values[:, numerator]
                    / _protected_denominator(values[:, denominator], epsilon),
                    f"({variable_names[numerator]})/protected({variable_names[denominator]}, eps={epsilon:.6g})",
                )
    if not columns:
        raise ValueError("The symbolic function library is empty")
    return np.column_stack(columns), names


def _symbolic_term_dependencies(
    n_variables: int,
    functions: list[str] | tuple[str, ...],
) -> list[list[int]]:
    enabled = list(dict.fromkeys(functions))
    dependencies: list[list[int]] = []
    unary = [name for name in enabled if name not in {"product", "ratio"}]
    for idx in range(n_variables):
        dependencies.extend([[idx] for _ in unary])
    if "product" in enabled:
        for left in range(n_variables):
            for right in range(left + 1, n_variables):
                dependencies.append([left, right])
    if "ratio" in enabled:
        for numerator in range(n_variables):
            for denominator in range(n_variables):
                if numerator != denominator:
                    dependencies.append([numerator, denominator])
    return dependencies


def _polynomial_term_dependencies(n_variables: int, degree: int) -> list[list[int]]:
    dependencies = [[idx] for idx in range(n_variables)]
    if degree >= 2:
        for left in range(n_variables):
            for right in range(left, n_variables):
                dependencies.append(sorted({left, right}))
    return dependencies


def _fit_sparse_library(
    library: np.ndarray,
    target: np.ndarray,
    term_names: list[str],
    max_terms: int,
) -> dict[str, Any]:
    """Select a compact term set with OMP and a BIC stopping rule, then refit."""

    library = np.asarray(library, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    if library.shape[0] != len(target) or library.shape[1] != len(term_names):
        raise ValueError("Symbolic library and target have incompatible shapes")
    standard_deviation = library.std(axis=0)
    usable = np.flatnonzero(np.isfinite(standard_deviation) & (standard_deviation > 1e-12))
    if len(usable) == 0:
        raise ValueError("The symbolic library contains no nonconstant finite terms")
    centered = library[:, usable] - library[:, usable].mean(axis=0, keepdims=True)
    normalized = centered / standard_deviation[usable]
    normalized /= np.maximum(np.linalg.norm(normalized, axis=0, keepdims=True), 1e-12)

    limit = min(max(1, int(max_terms)), len(usable), max(1, len(target) - 2))
    selected_local: list[int] = []
    residual = target - target.mean()
    best_selected: list[int] = []
    best_bic = float("inf")
    best_mse = float("inf")
    for _ in range(limit):
        scores = np.abs(normalized.T @ residual)
        if selected_local:
            scores[np.asarray(selected_local, dtype=int)] = -np.inf
        candidate = int(np.argmax(scores))
        if not np.isfinite(scores[candidate]):
            break
        selected_local.append(candidate)
        selected_global = usable[np.asarray(selected_local, dtype=int)]
        design = np.column_stack([np.ones(len(target)), library[:, selected_global]])
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ coefficients
        mse = float(np.mean(residual**2))
        bic = len(target) * np.log(max(mse, 1e-24)) + len(selected_local) * np.log(len(target))
        if bic < best_bic - 1e-10:
            best_bic = bic
            best_mse = mse
            best_selected = list(selected_local)
        if mse <= 1e-24:
            break

    if not best_selected:
        best_selected = [selected_local[0]]
    selected = usable[np.asarray(best_selected, dtype=int)]
    design = np.column_stack([np.ones(len(target)), library[:, selected]])
    sparse_coef = np.linalg.lstsq(design, target, rcond=None)[0]
    return {
        "intercept": float(sparse_coef[0]),
        "coefficients": [float(value) for value in sparse_coef[1:]],
        "term_indices": [int(value) for value in selected],
        "term_names": [term_names[int(value)] for value in selected],
        "selection_bic": float(best_bic),
        "fit_mse": float(best_mse),
    }


def _fit_sparse_polynomial(
    values: np.ndarray,
    target: np.ndarray,
    variable_names: list[str],
    degree: int,
    max_terms: int,
) -> dict[str, Any]:
    library, term_names = _polynomial_library(values, variable_names, degree)
    if library.shape[1] == 0:
        raise ValueError("A simple formula needs at least one input")
    design = np.column_stack([np.ones(len(library)), library])
    ridge = 1e-8 * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    try:
        dense_coef = np.linalg.solve(design.T @ design + ridge, design.T @ target)
    except np.linalg.LinAlgError:
        dense_coef = np.linalg.lstsq(design, target, rcond=None)[0]
    contribution = np.abs(dense_coef[1:]) * np.maximum(library.std(axis=0), 1e-12)
    term_count = min(max(1, int(max_terms)), library.shape[1], max(1, len(target) - 1))
    selected = np.argsort(contribution)[-term_count:]
    selected = selected[np.argsort(selected)]
    sparse_design = np.column_stack([np.ones(len(library)), library[:, selected]])
    sparse_coef = np.linalg.lstsq(sparse_design, target, rcond=None)[0]
    dependencies = _polynomial_term_dependencies(values.shape[1], degree)
    active_variables = sorted(
        {variable for term_idx in selected for variable in dependencies[int(term_idx)]}
    )
    return {
        "intercept": float(sparse_coef[0]),
        "coefficients": [float(value) for value in sparse_coef[1:]],
        "term_indices": [int(value) for value in selected],
        "term_names": [term_names[int(value)] for value in selected],
        "degree": int(degree),
        "method": "polynomial",
        "active_variable_indices": active_variables,
    }


def _fit_sparse_symbolic(
    values: np.ndarray,
    target: np.ndarray,
    variable_names: list[str],
    functions: list[str] | tuple[str, ...],
    epsilon: float,
    exp_clip: float,
    max_terms: int,
) -> dict[str, Any]:
    library, term_names = _symbolic_library(
        values,
        variable_names,
        functions,
        epsilon,
        exp_clip,
    )
    specification = _fit_sparse_library(library, target, term_names, max_terms)
    dependencies = _symbolic_term_dependencies(values.shape[1], functions)
    specification["active_variable_indices"] = sorted(
        {
            variable
            for term_idx in specification["term_indices"]
            for variable in dependencies[int(term_idx)]
        }
    )
    specification.update(
        {
            "method": "symbolic",
            "functions": list(dict.fromkeys(functions)),
            "epsilon": float(epsilon),
            "exp_clip": float(exp_clip),
        }
    )
    return specification


def _predict_sparse_polynomial(
    specification: dict[str, Any],
    values: np.ndarray,
    variable_names: list[str],
) -> np.ndarray:
    if specification.get("method", "polynomial") == "symbolic":
        library, _ = _symbolic_library(
            values,
            variable_names,
            specification["functions"],
            float(specification["epsilon"]),
            float(specification["exp_clip"]),
        )
    else:
        library, _ = _polynomial_library(values, variable_names, int(specification["degree"]))
    selected = np.asarray(specification["term_indices"], dtype=int)
    coefficients = np.asarray(specification["coefficients"], dtype=float)
    return float(specification["intercept"]) + library[:, selected] @ coefficients


def _absolute_correlation(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered_target = target - np.mean(target)
    target_norm = np.sqrt(np.sum(centered_target**2))
    centered_values = values - np.mean(values, axis=0, keepdims=True)
    value_norm = np.sqrt(np.sum(centered_values**2, axis=0))
    denominator = value_norm * target_norm
    numerator = np.abs(centered_values.T @ centered_target)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 1e-12,
    )


def _symbolic_feature_relevance(
    values: np.ndarray,
    target: np.ndarray,
    functions: list[str] | tuple[str, ...],
    epsilon: float,
    exp_clip: float,
) -> np.ndarray:
    unary_functions = [name for name in functions if name not in {"product", "ratio"}]
    relevance = np.zeros(values.shape[1], dtype=float)
    for idx in range(values.shape[1]):
        library, _ = _symbolic_library(
            values[:, [idx]],
            ["z"],
            unary_functions,
            epsilon,
            exp_clip,
        )
        relevance[idx] = float(np.max(_absolute_correlation(library, target)))
    return relevance


def _select_simple_formula(
    features: np.ndarray,
    teacher_target: np.ndarray,
    feature_names: list[str],
    min_inputs: int,
    max_inputs: int,
    max_terms: int,
    degree: int,
    seed: int,
    method: str = "polynomial",
    symbolic_functions: list[str] | tuple[str, ...] | None = None,
    epsilon: float = 1e-3,
    exp_clip: float = 8.0,
) -> dict[str, Any]:
    if len(features) < 8:
        raise ValueError("At least 8 teacher-training samples are required for formula distillation")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(features))
    selection_size = max(2, int(round(0.2 * len(features))))
    selection_size = min(selection_size, len(features) - 4)
    selection_indices = permutation[:selection_size]
    fit_indices = permutation[selection_size:]
    functions = list(symbolic_functions or DEFAULT_SYMBOLIC_FUNCTIONS)
    if method == "symbolic":
        correlations = _symbolic_feature_relevance(
            features[fit_indices],
            teacher_target[fit_indices],
            functions,
            epsilon,
            exp_clip,
        )
    elif method == "polynomial":
        correlations = _absolute_correlation(features[fit_indices], teacher_target[fit_indices])
    else:
        raise ValueError(f"Unsupported simple formula method {method!r}")
    pool_size = min(features.shape[1], max(20, 4 * max_inputs))
    candidate_pool = [int(value) for value in np.argsort(correlations)[-pool_size:][::-1]]

    selected: list[int] = []
    candidates_by_size: list[dict[str, Any]] = []
    for _ in range(min(max_inputs, len(candidate_pool))):
        best_candidate = None
        best_specification = None
        best_mae = float("inf")
        for candidate in candidate_pool:
            if candidate in selected:
                continue
            attempted = selected + [candidate]
            variable_names = [f"z{idx}" for idx in range(len(attempted))]
            if method == "symbolic":
                specification = _fit_sparse_symbolic(
                    features[fit_indices][:, attempted],
                    teacher_target[fit_indices],
                    variable_names,
                    functions,
                    epsilon,
                    exp_clip,
                    max_terms,
                )
            else:
                specification = _fit_sparse_polynomial(
                    features[fit_indices][:, attempted],
                    teacher_target[fit_indices],
                    variable_names,
                    degree,
                    max_terms,
                )
            prediction = _predict_sparse_polynomial(
                specification,
                features[selection_indices][:, attempted],
                variable_names,
            )
            selection_target = teacher_target[selection_indices]
            mae = float(np.mean(np.abs(prediction - selection_target)))
            denominator = float(
                np.sum((selection_target - np.mean(selection_target)) ** 2)
            )
            fidelity_r2 = (
                1.0 - float(np.sum((selection_target - prediction) ** 2)) / denominator
                if denominator > 1e-12
                else float("nan")
            )
            if mae < best_mae:
                best_mae = mae
                best_candidate = candidate
                best_specification = specification
                best_fidelity_r2 = fidelity_r2
        if best_candidate is None or best_specification is None:
            break
        selected.append(best_candidate)
        candidates_by_size.append(
            {
                "feature_indices": list(selected),
                "selection_teacher_mae": best_mae,
                "selection_teacher_r2": best_fidelity_r2,
            }
        )
    if not candidates_by_size:
        raise RuntimeError("No simple formula candidate could be fitted")

    eligible_by_size = [
        item for item in candidates_by_size if len(item["feature_indices"]) >= min_inputs
    ]
    if not eligible_by_size:
        raise ValueError(
            f"Could not fit a formula with at least {min_inputs} inputs; "
            f"only {len(candidates_by_size)} input steps were available"
        )
    minimum_mae = min(item["selection_teacher_mae"] for item in eligible_by_size)
    tolerance = minimum_mae * 1.02 + 1e-12
    chosen = next(
        item for item in eligible_by_size if item["selection_teacher_mae"] <= tolerance
    )
    selected = list(chosen["feature_indices"])
    variable_names = [f"z{idx}" for idx in range(len(selected))]
    if method == "symbolic":
        specification = _fit_sparse_symbolic(
            features[:, selected],
            teacher_target,
            variable_names,
            functions,
            epsilon,
            exp_clip,
            max_terms,
        )
    else:
        specification = _fit_sparse_polynomial(
            features[:, selected],
            teacher_target,
            variable_names,
            degree,
            max_terms,
        )
    specification.update(
        {
            "feature_indices": selected,
            "feature_names": [str(feature_names[idx]) for idx in selected],
            "active_feature_indices": [
                selected[idx]
                for idx in specification.get("active_variable_indices", range(len(selected)))
            ],
            "active_feature_names": [
                str(feature_names[selected[idx]])
                for idx in specification.get("active_variable_indices", range(len(selected)))
            ],
            "variable_names": variable_names,
            "selection_teacher_mae": float(chosen["selection_teacher_mae"]),
            "input_fidelity_curve": [
                {
                    "n_inputs": len(item["feature_indices"]),
                    "selection_teacher_mae": float(item["selection_teacher_mae"]),
                    "selection_teacher_r2_pct": 100.0
                    * float(item["selection_teacher_r2"]),
                }
                for item in eligible_by_size
            ],
        }
    )
    return specification


def _conformal_radius(residuals: np.ndarray, coverage: float) -> float:
    residuals = np.sort(np.asarray(residuals, dtype=float).reshape(-1))
    if not 0.0 < coverage < 1.0:
        raise ValueError("--simple-formula-coverage must be in (0, 1)")
    if len(residuals) == 0:
        raise ValueError("Formula fidelity calibration set is empty")
    rank = int(np.ceil((len(residuals) + 1) * coverage))
    if rank > len(residuals):
        return float("inf")
    return float(residuals[rank - 1])


def _formula_expression(specification: dict[str, Any]) -> str:
    terms = [f"{float(specification['intercept']):.12g}"]
    for coefficient, name in zip(
        specification["coefficients"], specification["term_names"]
    ):
        terms.append(f"{float(coefficient):+.12g}*{name}")
    return " ".join(terms)


def _formula_variable_definitions(
    prepared: dict[str, Any],
    selected_indices: list[int],
    variable_names: list[str],
) -> list[dict[str, Any]]:
    feature_names = list(prepared["selected_features"])
    pipeline = prepared.get("feature_pipeline")
    definitions = []
    for variable, feature_index in zip(variable_names, selected_indices):
        feature = str(feature_names[feature_index])
        definition: dict[str, Any] = {
            "variable": variable,
            "feature": feature,
            "expression": f"preprocessed({feature!r})",
        }
        if pipeline is not None:
            imputer = pipeline.named_steps.get("imputer")
            statistic = (
                float(imputer.statistics_[feature_index])
                if imputer is not None and hasattr(imputer, "statistics_")
                else float("nan")
            )
            definition["impute_value"] = statistic
            imputed = f"impute({feature!r}, {statistic:.12g})"
            scaler = pipeline.named_steps.get("scaler")
            if isinstance(scaler, MinMaxScaler):
                scale = float(scaler.scale_[feature_index])
                offset = float(scaler.min_[feature_index])
                definition.update({"scale": scale, "offset": offset})
                definition["expression"] = f"{scale:.12g}*{imputed}{offset:+.12g}"
            elif isinstance(scaler, StandardScaler):
                center = float(scaler.mean_[feature_index])
                scale = float(scaler.scale_[feature_index])
                definition.update({"center": center, "scale": scale})
                definition["expression"] = (
                    f"({imputed}-{center:.12g})/{scale:.12g}"
                )
            else:
                definition["expression"] = imputed
        definitions.append(definition)
    return definitions


def distill_simple_formulas(
    model: torch.nn.Module,
    prepared: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    task_type: str,
    target_names: list[str],
    model_label: str,
    fold: int,
) -> dict[str, Any]:
    if task_type != "regression":
        raise ValueError("Simple symbolic formula distillation currently supports regression only")
    min_inputs = int(args.simple_formula_min_inputs)
    max_inputs = int(args.simple_formula_max_inputs)
    max_terms = int(args.simple_formula_max_terms)
    if min_inputs < 1 or max_inputs < min_inputs or max_terms < 1:
        raise ValueError("Simple formula input and term limits must be positive")
    if len(prepared["x_calibration"]) == 0:
        raise ValueError("No reserved formula calibration set is available")

    formula_train_features = prepared.get("x_formula_train", prepared["x_train"])
    teacher_train = predict_features(
        model,
        formula_train_features,
        prepared["target_scaler"],
        device,
        task_type,
        args.batch_size,
    )
    teacher_calibration = predict_features(
        model,
        prepared["x_calibration"],
        prepared["target_scaler"],
        device,
        task_type,
        args.batch_size,
    )
    teacher_test = predict_features(
        model,
        prepared["x_test"],
        prepared["target_scaler"],
        device,
        task_type,
        args.batch_size,
    )
    target_test = np.asarray(prepared["y_test"], dtype=float).reshape(len(teacher_test), -1)
    records = []
    text_lines = [
        f"Simple post-hoc surrogate for {model_label}",
        f"dataset = {args.dataset}",
        f"outer_fold = {fold}",
        "teacher_role = exact outer benchmark model instance",
        "teacher_training_scope = full outer train+validation partition",
        "scope = formula distillation and calibration use only the outer train+validation partition",
        "formula_split = formula fitting and calibration are disjoint; the fitted teacher may have seen calibration inputs during benchmark training",
        "outer_test_role = final evaluation only",
        f"requested_conformal_coverage = {100.0 * args.simple_formula_coverage:.6g}%",
        f"formula_method = {args.simple_formula_method}",
        "symbolic_functions = "
        + ",".join(args.simple_formula_functions)
        if args.simple_formula_method == "symbolic"
        else f"polynomial_degree = {args.simple_formula_degree}",
        "coverage_statement = P(|teacher(x)-formula(x)| <= radius) >= requested coverage under exchangeability",
        "note = this is an assumption-conditioned split-conformal statement, not an unconditional guarantee",
        "note = exchangeability must remain credible despite reuse of formula-calibration inputs in teacher training",
        "note = the coverage percentage is meaningful only together with its target-unit radius",
        "",
    ]
    for target_idx, target_name in enumerate(target_names):
        specification = _select_simple_formula(
            formula_train_features,
            teacher_train[:, target_idx],
            list(prepared["selected_features"]),
            min_inputs=min_inputs,
            max_inputs=max_inputs,
            max_terms=max_terms,
            degree=int(args.simple_formula_degree),
            seed=args.seed + fold * 1009 + target_idx,
            method=args.simple_formula_method,
            symbolic_functions=args.simple_formula_functions,
            epsilon=float(args.simple_formula_epsilon),
            exp_clip=float(args.simple_formula_exp_clip),
        )
        selected = specification["feature_indices"]
        variable_names = specification["variable_names"]
        formula_calibration = _predict_sparse_polynomial(
            specification,
            prepared["x_calibration"][:, selected],
            variable_names,
        )
        calibration_residual = np.abs(
            teacher_calibration[:, target_idx] - formula_calibration
        )
        radius = _conformal_radius(calibration_residual, args.simple_formula_coverage)
        formula_test = _predict_sparse_polynomial(
            specification,
            prepared["x_test"][:, selected],
            variable_names,
        )
        teacher_residual = np.abs(teacher_test[:, target_idx] - formula_test)
        teacher_mae = float(np.mean(teacher_residual))
        teacher_target_mae = float(
            np.mean(np.abs(target_test[:, target_idx] - teacher_test[:, target_idx]))
        )
        target_mae = float(np.mean(np.abs(target_test[:, target_idx] - formula_test)))
        empirical_coverage = float(np.mean(teacher_residual <= radius))
        denominator = float(np.sum((teacher_test[:, target_idx] - np.mean(teacher_test[:, target_idx])) ** 2))
        fidelity_r2 = (
            1.0 - float(np.sum((teacher_test[:, target_idx] - formula_test) ** 2)) / denominator
            if denominator > 1e-12
            else float("nan")
        )
        record = {
            **specification,
            "target": target_name,
            "expression": _formula_expression(specification),
            "formula_method": args.simple_formula_method,
            "conformal_coverage": float(args.simple_formula_coverage),
            "conformal_radius": radius,
            "calibration_size": int(len(calibration_residual)),
            "test_teacher_mae": teacher_mae,
            "test_teacher_target_mae": teacher_target_mae,
            "test_target_mae": target_mae,
            "test_fidelity_r2": fidelity_r2,
            "test_empirical_coverage": empirical_coverage,
        }
        active_variables = specification.get(
            "active_variable_indices", list(range(len(selected)))
        )
        record["variable_definitions"] = _formula_variable_definitions(
            prepared,
            [selected[idx] for idx in active_variables],
            [variable_names[idx] for idx in active_variables],
        )
        records.append(record)
        text_lines.extend(
            [
                f"target = {target_name}",
                *[
                    f"  {definition['variable']} = {definition['expression']}"
                    for definition in record["variable_definitions"]
                ],
                f"  formula = {record['expression']}",
                f"  formula_method = {args.simple_formula_method}",
                f"  active_inputs = {len(active_variables)}",
                f"  searched_inputs = {len(selected)}",
                f"  nonconstant_terms = {len(specification['coefficients'])}",
                "  validation_fidelity_curve = "
                + ", ".join(
                    f"{item['n_inputs']} inputs: MAE {item['selection_teacher_mae']:.12g}, "
                    f"R2 {item['selection_teacher_r2_pct']:.6g}%"
                    for item in specification["input_fidelity_curve"]
                ),
                f"  teacher_to_target_test_MAE = {teacher_target_mae:.12g}",
                f"  formula_to_teacher_test_MAE = {teacher_mae:.12g}",
                f"  formula_to_target_test_MAE = {target_mae:.12g}",
                f"  formula_to_teacher_test_R2 = {fidelity_r2:.12g}",
                f"  conformal_radius = {radius:.12g}",
                f"  empirical_outer_test_coverage = {100.0 * empirical_coverage:.6g}%",
                "",
            ]
        )

    output_base = Path(args.output_dir) / (
        f"simple-formula-{args.dataset}-fold{fold}-{model_label}"
    )
    text_path = output_base.with_suffix(".txt")
    json_path = output_base.with_suffix(".json")
    text_path.write_text("\n".join(text_lines), encoding="utf-8")
    json_path.write_text(json.dumps({"targets": records}, indent=2), encoding="utf-8")
    return {
        "simple_formula_path": str(text_path),
        "simple_formula_json_path": str(json_path),
        "simple_formula_n_inputs": max(
            len(record["active_feature_indices"]) for record in records
        ),
        "simple_formula_n_searched_inputs": max(
            len(record["feature_indices"]) for record in records
        ),
        "simple_formula_n_terms": max(len(record["coefficients"]) for record in records),
        "simple_formula_method": args.simple_formula_method,
        "simple_formula_functions": (
            ",".join(args.simple_formula_functions)
            if args.simple_formula_method == "symbolic"
            else f"degree_{args.simple_formula_degree}_polynomial"
        ),
        "simple_formula_inputs": ";".join(
            ",".join(record["active_feature_names"]) for record in records
        ),
        "simple_formula_requested_coverage_pct": 100.0 * float(args.simple_formula_coverage),
        "simple_formula_conformal_radius": float(
            np.mean([record["conformal_radius"] for record in records])
        ),
        "simple_formula_test_empirical_coverage_pct": 100.0
        * float(np.mean([record["test_empirical_coverage"] for record in records])),
        "simple_formula_test_teacher_mae": float(
            np.mean([record["test_teacher_mae"] for record in records])
        ),
        "simple_formula_test_target_mae": float(
            np.mean([record["test_target_mae"] for record in records])
        ),
        "simple_formula_test_fidelity_r2_pct": 100.0
        * float(np.mean([record["test_fidelity_r2"] for record in records])),
        "simple_formula_calibration_size": min(record["calibration_size"] for record in records),
    }


def run_model(
    model_name: str,
    fold_data: dict[str, Any],
    prepared: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float | int | str], list[Any]]:
    set_seed(args.seed)
    model_label = canonical_model_name(model_name, args)
    common_dims, group_dims, property_dims, target_dims = model_dims_for_family(model_label, args)
    task_type = str(fold_data["metadata"]["task_type"])
    target_names = target_names_from_metadata(fold_data["metadata"])
    target_name = target_display_name(target_names)
    model = build_model(
        model_name,
        n_feat=prepared["x_train"].shape[1],
        target_names=target_names,
        args=args,
    ).to(device)
    optimizer = make_optimizer(model, args)
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
    best_train_loss = float("inf")
    epochs_ran = 0
    print(
        f"\n[{model_label}] fold {fold_data['metadata']['fold']} start: "
        f"features={prepared['x_train'].shape[1]}, epochs={args.epochs}",
        flush=True,
    )
    sync(device)
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        if isinstance(model, SymbolicKAN):
            model.set_progress(
                (epoch - 1) / max(args.epochs - 1, 1)
            )
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            task_type,
            args.loss,
            kan_l1_lambda=args.kan_l1_lambda if is_kan_family(model_label) else 0.0,
            kan_sparsity_mode=args.kan_sparsity_mode,
        )
        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                prepared["target_scaler"],
                device,
                task_type,
                target_names=target_names,
            )
            improved = metric_is_better(val_metrics[val_metric_name], best_metric, task_type)
            if improved:
                best_metric = val_metrics[val_metric_name]
                best_val_metrics = val_metrics
                best_epoch = epoch
                if args.restore_best_state:
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

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
        elif args.log_every_epochs > 0 and (
            epoch % args.log_every_epochs == 0 or epoch == 1 or epoch == args.epochs
        ):
            print(f"[{model_label}] epoch {epoch}/{args.epochs} train_loss={train_loss:.6g}", flush=True)

        monitor_improved = False
        if args.early_stopping_monitor == "validation" and val_loader is not None:
            monitor_improved = improved
        elif args.early_stopping_monitor == "validation":
            # A final fit with val_ratio=0 has no validation signal, so keep
            # the historical fixed-epoch behaviour for this monitor.
            monitor_improved = True
        elif args.early_stopping_monitor == "loss":
            monitor_improved = train_loss < (
                best_train_loss - float(args.early_stopping_min_delta)
            )
            if monitor_improved:
                best_train_loss = train_loss
        elif args.early_stopping_monitor == "none":
            monitor_improved = True

        if monitor_improved:
            patience_left = args.early_stopping_patience
        elif args.early_stopping_monitor != "none":
            patience_left -= 1
        if (
            args.early_stopping_monitor != "none"
            and args.early_stopping_patience > 0
            and patience_left <= 0
        ):
            print(
                f"[{model_label}] early stop at epoch {epoch} "
                f"(monitor={args.early_stopping_monitor})",
                flush=True,
            )
            break
    sync(device)
    train_seconds = time.perf_counter() - train_start

    if args.restore_best_state and best_state is not None:
        model.load_state_dict(best_state)
    elif val_loader is not None:
        # MODNet fit_preset uses restore_best_weights=False and ranks the model
        # left at the end of loss-based early stopping on held-out data.
        best_val_metrics = evaluate(
            model,
            val_loader,
            prepared["target_scaler"],
            device,
            task_type,
            target_names=target_names,
        )
        best_epoch = epochs_ran
    params = count_parameters(model)
    pruning_masks = PruningMasks()
    prune_finetune_seconds = 0.0
    if is_kan_family(model_label) and args.prune_kan_fraction > 0:
        pruning_masks = apply_kan_pruning(model, args.prune_kan_fraction, args.prune_mode)
        print(
            f"[{model_label}] pruned {pruning_masks.pruned_parameters} KAN parameters"
            + (
                f" across {pruning_masks.pruned_edges}/{pruning_masks.total_edges} edges"
                if args.prune_mode == "edge"
                else ""
            ),
            flush=True,
        )
        if args.prune_finetune_epochs > 0:
            finetune_optimizer = (
                torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
                if args.weight_decay == 0
                else torch.optim.AdamW(
                    adamw_parameter_groups(model, args.weight_decay),
                    lr=args.lr * 0.1,
                )
            )
            prune_best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            prune_best_metrics = (
                evaluate(
                    model,
                    val_loader,
                    prepared["target_scaler"],
                    device,
                    task_type,
                    target_names=target_names,
                )
                if val_loader is not None
                else best_val_metrics
            )
            prune_best_metric = prune_best_metrics.get(val_metric_name, float("nan"))
            finetune_start = time.perf_counter()
            for finetune_epoch in range(1, args.prune_finetune_epochs + 1):
                train_one_epoch(
                    model,
                    train_loader,
                    finetune_optimizer,
                    device,
                    task_type,
                    args.loss,
                    kan_l1_lambda=args.kan_l1_lambda,
                    kan_sparsity_mode=args.kan_sparsity_mode,
                    pruning_masks=pruning_masks,
                )
                if val_loader is not None:
                    candidate_metrics = evaluate(
                        model,
                        val_loader,
                        prepared["target_scaler"],
                        device,
                        task_type,
                        target_names=target_names,
                    )
                    if metric_is_better(
                        candidate_metrics[val_metric_name], prune_best_metric, task_type
                    ):
                        prune_best_metric = candidate_metrics[val_metric_name]
                        prune_best_metrics = candidate_metrics
                        prune_best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                if args.log_every_epochs > 0 and (
                    finetune_epoch == 1
                    or finetune_epoch == args.prune_finetune_epochs
                    or finetune_epoch % args.log_every_epochs == 0
                ):
                    print(
                        f"[{model_label}] prune fine-tune "
                        f"{finetune_epoch}/{args.prune_finetune_epochs}",
                        flush=True,
                    )
            sync(device)
            prune_finetune_seconds = time.perf_counter() - finetune_start
            model.load_state_dict(prune_best_state)
            pruning_masks.enforce()
            best_val_metrics = prune_best_metrics
        if val_loader is not None:
            best_val_metrics = evaluate(
                model,
                val_loader,
                prepared["target_scaler"],
                device,
                task_type,
                target_names=target_names,
            )
    pruned_params = pruning_masks.pruned_parameters
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
        test_metrics = metrics_from_predictions(
            test_prediction,
            test_target,
            task_type,
            target_names=target_names,
        )
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
        "inner_fold": args.inner_fold_index if args.inner_fold_index is not None else "",
        "inner_n_splits": args.inner_n_splits if args.inner_fold_index is not None else "",
        "task_type": task_type,
        "target": target_name,
        "target_names": ";".join(target_names),
        "n_targets": len(target_names),
        "model": model_label,
        "architecture": getattr(model, "architecture", "unknown"),
        "block_type": "mlp" if model_label == "mlp" else "hybrid" if model_label.startswith("hybrid-") else "kan",
        "kan_impl": (
            "symbolic-primitives"
            if model_label == "symbolic-kan"
            else kan_impl_for_family(model_label, args.kan_impl)
            if is_kan_family(model_label)
            else "none"
        ),
        "kan_grid_size": (
            "none" if model_label == "symbolic-kan"
            else args.kan_grid_size if is_kan_family(model_label) else "none"
        ),
        "kan_spline_order": (
            args.kan_spline_order
            if kan_impl_for_family(model_label, args.kan_impl) == "spline"
            and model_label != "symbolic-kan"
            else "none"
        ),
        "common_dims": "-".join(str(value) for value in common_dims),
        "group_dims": "-".join(str(value) for value in group_dims),
        "property_dims": "-".join(str(value) for value in property_dims),
        "target_dims": "-".join(str(value) for value in target_dims),
        "featurizer_preset": prepared["feature_preset"],
        "n_features": prepared["x_train"].shape[1],
        "params": params,
        "effective_params": effective_params,
        "pruned_params": pruned_params,
        "params_before_prune": params,
        "params_after_prune": effective_params,
        "params_pruned": pruned_params,
        "params_pruned_pct": 100.0 * pruned_params / params if params else 0.0,
        "prune_kan_fraction": args.prune_kan_fraction if is_kan_family(model_label) else 0.0,
        "prune_mode": args.prune_mode if is_kan_family(model_label) else "none",
        "pruned_edges": pruning_masks.pruned_edges,
        "total_kan_edges": pruning_masks.total_edges,
        "prune_finetune_epochs": args.prune_finetune_epochs if is_kan_family(model_label) else 0,
        "prune_finetune_seconds": prune_finetune_seconds,
        "kan_l1_lambda": args.kan_l1_lambda if is_kan_family(model_label) else 0.0,
        "kan_sparsity_lambda": (
            args.kan_l1_lambda if is_kan_family(model_label) else 0.0
        ),
        "kan_sparsity_mode": (
            args.kan_sparsity_mode
            if is_kan_family(model_label) and args.kan_l1_lambda > 0
            else "none"
        ),
        "activation": args.activation,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "max_epochs": args.epochs,
        "early_stopping_monitor": args.early_stopping_monitor,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "early_stopping_patience": args.early_stopping_patience,
        "restore_best_state": args.restore_best_state,
        "train_seconds": train_seconds,
        "forward_ms_per_batch": forward_ms,
    }
    if model_label == "symbolic-kan":
        row.update(
            {
                "symbolic_hidden_dims": "-".join(
                    str(value) for value in args.symbolic_hidden_dims
                ),
                "symbolic_edges_per_unit": args.symbolic_edges_per_unit,
                "symbolic_primitives": ",".join(args.symbolic_primitives),
                "symbolic_temperature_start": args.symbolic_temperature_start,
                "symbolic_temperature_end": args.symbolic_temperature_end,
                "symbolic_projection_top_k": args.symbolic_projection_top_k,
            }
        )
    for metric_name, value in best_val_metrics.items():
        row[f"best_val_{metric_name}"] = value
    for metric_name, value in test_metrics.items():
        row[f"test_{metric_name}"] = value
    if (
        args.export_formulas
        and is_kan_family(model_label)
        and model_label != "symbolic-kan"
    ):
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
            target_name=target_name,
            task_type=task_type,
            target_scaler=prepared["target_scaler"],
            params_before_prune=params,
            params_after_prune=effective_params,
            pruned_params=pruned_params,
            prune_fraction=args.prune_kan_fraction,
            best_val_metrics=best_val_metrics,
            test_metrics=test_metrics,
            fold_metadata=fold_data["metadata"],
        )
        row["formula_path"] = str(formula_path)
        if model_label == "direct-spline":
            row["spline_exact_formula_test_mae"] = test_metrics["mae"]
    if args.symbolify_spline_kan and model_label == "direct-spline":
        if task_type != "regression" or test_loader is None:
            raise ValueError(
                "Spline KAN auto-symbolic evaluation requires a regression outer test fold"
            )
        row.update(
            symbolify_and_evaluate_spline_kan(
                model,
                prepared,
                args,
                target_names,
                test_prediction,
                test_target,
                int(fold_data["metadata"]["fold"]),
            )
        )
    if model_label == "symbolic-kan":
        if test_loader is None:
            raise ValueError("Symbolic-KAN hard formula evaluation requires the outer test fold")
        row.update(
            harden_and_evaluate_symbolic_kan(
                model,
                train_loader,
                test_loader,
                prepared,
                args,
                device,
                task_type,
                target_names,
                test_prediction,
                test_target,
                int(fold_data["metadata"]["fold"]),
            )
        )
    if args.distill_simple_formula:
        row.update(
            distill_simple_formulas(
                model,
                prepared,
                args,
                device,
                task_type,
                target_names,
                model_label,
                int(fold_data["metadata"]["fold"]),
            )
        )
    if len(test_prediction) == 0:
        predictions = []
    else:
        prediction_array = test_prediction.numpy().reshape(test_prediction.shape[0], -1)
        predictions = prediction_array[:, 0].tolist() if prediction_array.shape[1] == 1 else prediction_array.tolist()
    pruning_masks.remove_hooks()
    return row, predictions


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

    teacher_train_indices, val_indices = split_train_val(
        len(train_features),
        args.val_ratio,
        args.seed,
        targets=fold_data["train_targets"],
        task_type=fold_data["metadata"]["task_type"],
        inner_fold_index=args.inner_fold_index,
        inner_n_splits=args.inner_n_splits,
    )
    train_targets = as_2d_targets(fold_data["train_targets"])
    test_targets = as_2d_targets(fold_data["test_targets"])
    formula_train_indices, calibration_indices = reserve_formula_calibration(
        teacher_train_indices,
        train_targets,
        args,
        str(fold_data["metadata"]["task_type"]),
    )
    processor = MODNetFeatureProcessor(
        n_features=args.n_features,
        scaler=args.scaler,
        impute_strategy=args.impute_strategy,
        random_state=args.seed,
        task_type=fold_data["metadata"]["task_type"],
    )
    processor.fit(
        train_features.iloc[teacher_train_indices],
        feature_selection_target(train_targets[teacher_train_indices]),
    )

    x_all_train = processor.transform(train_features)
    x_test = (
        processor.transform(test_features)
        if test_features is not None
        else np.empty((0, x_all_train.shape[1]), dtype=np.float32)
    )
    y_train = train_targets[teacher_train_indices]
    y_val = train_targets[val_indices] if len(val_indices) else np.empty((0, train_targets.shape[1]), dtype=np.float32)
    target_scaler = TargetScaler(y_train, mode=args.target_scale)
    y_all_train_scaled = target_scaler.transform(train_targets)
    y_test_scaled = (
        target_scaler.transform(test_targets)
        if len(test_targets)
        else np.empty((0, train_targets.shape[1]), dtype=np.float32)
    )

    return {
        "feature_preset": actual_train_preset,
        "featurize_seconds": featurize_seconds,
        "processor": processor,
        "feature_pipeline": None,
        "selected_features": processor.selected_columns_,
        "target_scaler": target_scaler,
        "train_size": len(teacher_train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_targets),
        "x_train": x_all_train[teacher_train_indices],
        "y_train_scaled": y_all_train_scaled[teacher_train_indices],
        "x_formula_train": x_all_train[formula_train_indices],
        "x_val": x_all_train[val_indices] if len(val_indices) else np.empty((0, x_all_train.shape[1]), dtype=np.float32),
        "y_val_scaled": y_all_train_scaled[val_indices] if len(val_indices) else np.empty((0, train_targets.shape[1]), dtype=np.float32),
        "x_calibration": x_all_train[calibration_indices] if len(calibration_indices) else np.empty((0, x_all_train.shape[1]), dtype=np.float32),
        "y_calibration": train_targets[calibration_indices] if len(calibration_indices) else np.empty((0, train_targets.shape[1]), dtype=np.float32),
        "x_test": x_test,
        "y_test_scaled": y_test_scaled,
        "y_test": test_targets,
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

    teacher_train_indices, val_indices = split_train_val(
        len(train_features),
        args.val_ratio,
        args.seed,
        targets=fold_data["train_targets"],
        task_type=fold_data["metadata"]["task_type"],
        inner_fold_index=args.inner_fold_index,
        inner_n_splits=args.inner_n_splits,
    )
    train_targets = as_2d_targets(fold_data["train_targets"])
    test_targets = as_2d_targets(fold_data["test_targets"])
    formula_train_indices, calibration_indices = reserve_formula_calibration(
        teacher_train_indices,
        train_targets,
        args,
        str(fold_data["metadata"]["task_type"]),
    )

    pipeline = make_feature_pipeline(args)
    pipeline.fit(train_features.iloc[teacher_train_indices][selected_features])
    x_all_train = pipeline.transform(train_features[selected_features]).astype(np.float32, copy=False)
    x_test = pipeline.transform(test_features.reindex(columns=selected_features)).astype(np.float32, copy=False)

    y_train = train_targets[teacher_train_indices]
    y_val = train_targets[val_indices] if len(val_indices) else np.empty((0, train_targets.shape[1]), dtype=np.float32)
    target_scaler = TargetScaler(y_train, mode=args.target_scale)
    y_all_train_scaled = target_scaler.transform(train_targets)
    y_test_scaled = target_scaler.transform(test_targets) if len(test_targets) else np.empty((0, train_targets.shape[1]), dtype=np.float32)

    return {
        "feature_preset": "official-modnet-precomputed",
        "featurize_seconds": 0.0,
        "processor": None,
        "feature_pipeline": pipeline,
        "selected_features": selected_features,
        "target_scaler": target_scaler,
        "train_size": len(teacher_train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_targets),
        "x_train": x_all_train[teacher_train_indices],
        "y_train_scaled": y_all_train_scaled[teacher_train_indices],
        "x_formula_train": x_all_train[formula_train_indices],
        "x_val": x_all_train[val_indices] if len(val_indices) else np.empty((0, x_all_train.shape[1]), dtype=np.float32),
        "y_val_scaled": y_all_train_scaled[val_indices] if len(val_indices) else np.empty((0, train_targets.shape[1]), dtype=np.float32),
        "x_calibration": x_all_train[calibration_indices] if len(calibration_indices) else np.empty((0, x_all_train.shape[1]), dtype=np.float32),
        "y_calibration": train_targets[calibration_indices] if len(calibration_indices) else np.empty((0, train_targets.shape[1]), dtype=np.float32),
        "x_test": x_test,
        "y_test_scaled": y_test_scaled,
        "y_test": test_targets,
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
        simple_formula_paths = [
            str(row["simple_formula_path"])
            for row in model_rows
            if row.get("simple_formula_path")
        ]
        if simple_formula_paths:
            item["simple_formula_paths"] = ";".join(simple_formula_paths)
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
                item[f"{key}_mean"] = finite_mean([float(row.get(key, float("nan"))) for row in model_rows])
                item[f"{key}_std"] = finite_std([float(row.get(key, float("nan"))) for row in model_rows])
        dynamic_metric_keys = sorted(
            key
            for row in model_rows
            for key in row
            if key.startswith(("best_val_", "test_")) and key not in item
        )
        for key in dynamic_metric_keys:
            item[f"{key}_mean"] = finite_mean([float(row.get(key, float("nan"))) for row in model_rows])
            item[f"{key}_std"] = finite_std([float(row.get(key, float("nan"))) for row in model_rows])
        simple_numeric_keys = sorted(
            key
            for row in model_rows
            for key, value in row.items()
            if key.startswith("simple_formula_") and isinstance(value, (int, float))
        )
        for key in simple_numeric_keys:
            item[f"{key}_mean"] = finite_mean(
                [float(row.get(key, float("nan"))) for row in model_rows]
            )
            item[f"{key}_std"] = finite_std(
                [float(row.get(key, float("nan"))) for row in model_rows]
            )
        summary.append(item)
    return summary


def finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else float("nan")


def finite_std(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    return float(np.std(finite_values)) if finite_values else float("nan")


def _flatten_formula_module(
    name: str,
    module: torch.nn.Module,
) -> Iterable[tuple[str, torch.nn.Module]]:
    # KAN layers contain implementation Linear modules; they are one logical
    # operation and must not be traversed again or the exported equation is
    # overwritten by duplicate internal layers.
    if isinstance(
        module,
        (
            FastKANLinear,
            KANLinear,
            torch.nn.Linear,
            torch.nn.ReLU,
            torch.nn.ELU,
            torch.nn.SiLU,
            torch.nn.LayerNorm,
            torch.nn.BatchNorm1d,
        ),
    ):
        yield name, module
        return
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        yield from _flatten_formula_module(full_name, child)


def _formula_forward_modules(model: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Module]]:
    """Yield the actual single-property MODNet forward path in order."""

    output_heads = getattr(model, "output_heads", None)
    if output_heads is None or len(output_heads) != 1:
        raise ValueError("Formula export currently requires one MODNet property head")
    prop_key = next(iter(output_heads.keys()))
    group_key = prop_key.split("_p", 1)[0]
    path = (
        ("common_block", model.common_block),
        (f"group_blocks.{group_key}", model.group_blocks[group_key]),
        (f"property_blocks.{prop_key}", model.property_blocks[prop_key]),
        (f"target_blocks.{prop_key}", model.target_blocks[prop_key]),
        (f"output_heads.{prop_key}", model.output_heads[prop_key]),
    )
    for name, module in path:
        yield from _flatten_formula_module(name, module)


def _activation_formula_lines(
    name: str,
    module: torch.nn.Module,
    source_names: list[str],
    output_names: list[str],
    top_k: int,
) -> list[str]:
    lines = [f"  # {name}: {module.__class__.__name__}"]
    if isinstance(module, torch.nn.ReLU):
        lines.extend(
            f"  {output} = relu({_source_name(source_names, idx)})"
            for idx, output in enumerate(output_names)
        )
    elif isinstance(module, torch.nn.ELU):
        lines.extend(
            f"  {output} = elu({_source_name(source_names, idx)}; alpha={module.alpha:g})"
            for idx, output in enumerate(output_names)
        )
    elif isinstance(module, torch.nn.SiLU):
        lines.extend(
            f"  {output} = silu({_source_name(source_names, idx)})"
            for idx, output in enumerate(output_names)
        )
    elif isinstance(module, torch.nn.LayerNorm):
        gamma = (
            module.weight.detach().cpu().reshape(-1).tolist()
            if module.elementwise_affine
            else [1.0] * len(source_names)
        )
        beta = (
            module.bias.detach().cpu().reshape(-1).tolist()
            if module.elementwise_affine
            else [0.0] * len(source_names)
        )
        vector = _compact_names(source_names)
        for idx, output in enumerate(output_names):
            lines.append(
                f"  {output} = LayerNorm_{idx}({vector}; "
                f"gamma={_format_number(float(gamma[idx]))}, "
                f"beta={_format_number(float(beta[idx]))}, eps={module.eps:g})"
            )
    elif isinstance(module, torch.nn.BatchNorm1d):
        mean = module.running_mean.detach().cpu()
        var = module.running_var.detach().cpu()
        gamma = module.weight.detach().cpu() if module.affine else torch.ones_like(mean)
        beta = module.bias.detach().cpu() if module.affine else torch.zeros_like(mean)
        for idx, output in enumerate(output_names):
            scale = float(gamma[idx] / torch.sqrt(var[idx] + module.eps))
            offset = float(beta[idx] - mean[idx] * scale)
            lines.append(
                f"  {output} = {_format_number(scale)}*{_source_name(source_names, idx)} "
                f"+ {_format_number(offset)}"
            )
    lines.append("")
    return lines


def export_sparse_formula(
    model: torch.nn.Module,
    model_label: str,
    input_names: list[str] | None,
    output_path: Path,
    top_k: int,
    min_abs: float,
    target_name: str,
    task_type: str,
    target_scaler: TargetScaler,
    params_before_prune: int,
    params_after_prune: int,
    pruned_params: int,
    prune_fraction: float,
    best_val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    fold_metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_inputs = len(input_names or []) or int(getattr(model, "n_feat", 0))
    raw_input_names = input_names or [f"feature_{idx}" for idx in range(n_inputs)]
    input_vars = [f"x{idx}" for idx in range(n_inputs)]
    target_names = target_names_from_metadata(fold_metadata)
    safe_target = _safe_symbol(target_name)
    safe_target_names = [_safe_symbol(name) for name in target_names]
    formula_scope = (
        "exact: every nonzero term after pruning is written"
        if top_k <= 0
        else f"readable: top {top_k} nonzero terms per neuron after pruning are written"
    )
    lines = [
        f"Explicit pruned formula for {model_label}",
        f"target = {target_name}",
        formula_scope,
        f"min_abs = {min_abs:g}",
        f"params_before_prune = {params_before_prune}",
        f"params_after_prune = {params_after_prune}",
        f"params_pruned = {pruned_params}",
        f"requested_prune_fraction = {prune_fraction:g}",
        "",
        "Performance after pruning",
        f"  dataset = {fold_metadata.get('dataset', '')}",
        f"  fold = {fold_metadata.get('fold', '')}",
        f"  task_type = {task_type}",
        f"  train_and_val_size = {fold_metadata.get('train_and_val_size', '')}",
        f"  test_size = {fold_metadata.get('test_size', '')}",
    ]
    lines.extend(_formula_metric_lines("validation", best_val_metrics, task_type, target_names))
    lines.extend(_formula_metric_lines("test", test_metrics, task_type, target_names))
    lines.extend(
        [
            "  note = metrics are computed from the full pruned model; a top-k formula is a display-only truncation.",
            "  confidence_note = this is held-out performance and formula-fidelity, not calibrated uncertainty.",
            "",
        ]
    )
    lines.extend(
        [
        "Inputs",
        "  x_i are selected descriptors after the benchmark imputer/scaler.",
        "  Raw descriptor names:",
        ]
    )
    for idx, name in enumerate(raw_input_names):
        lines.append(f"    x{idx} = {_safe_symbol(str(name))}")
    lines.extend(
        [
            "",
            "Function definitions",
            "  relu(x) = max(0, x)",
            "  elu(x; alpha) = x if x > 0 else alpha*(exp(x)-1)",
            "  silu(x) = x / (1 + exp(-x))",
            "  rbf(x; c, h) = exp(-((x - c) / h)^2)",
            "  LayerNorm_i(v; gamma,beta,eps) = gamma_i * (v_i - mean(v)) / sqrt(var(v) + eps) + beta_i",
            "  spline_b{i,k}(x) means the kth B-spline basis for input i with the layer knots below.",
            "",
            "Equations",
        ]
    )

    formula_modules = list(_formula_forward_modules(model))
    transform_indices = [
        idx
        for idx, (_, module) in enumerate(formula_modules)
        if isinstance(module, (FastKANLinear, KANLinear, torch.nn.Linear))
    ]
    last_transform_index = transform_indices[-1] if transform_indices else -1
    source_names = input_vars
    final_output_names: list[str] = []
    coverage_values: list[float] = []
    for module_index, (name, module) in enumerate(formula_modules):
        is_final = module_index == last_transform_index
        if is_final:
            raw_name = "logit" if task_type == "classification" else "raw"
            out_features = int(getattr(module, "out_features"))
            output_names = [
                f"{raw_name}_{safe_target_names[idx] if idx < len(safe_target_names) else f'{safe_target}_{idx}'}"
                for idx in range(out_features)
            ]
        else:
            width = int(getattr(module, "out_features", len(source_names)))
            output_names = [f"z{module_index:02d}_{idx}" for idx in range(width)]
        if isinstance(module, FastKANLinear):
            lines.extend(
                _fastkan_formula_lines(
                    name,
                    module,
                    source_names,
                    output_names,
                    module_index,
                    top_k,
                    min_abs,
                    coverage_values,
                )
            )
        elif isinstance(module, KANLinear):
            lines.extend(
                _spline_formula_lines(
                    name,
                    module,
                    source_names,
                    output_names,
                    top_k,
                    min_abs,
                    coverage_values,
                )
            )
        elif isinstance(module, torch.nn.Linear):
            lines.extend(
                _linear_formula_lines(
                    name,
                    module,
                    source_names,
                    output_names,
                    top_k,
                    min_abs,
                    coverage_values,
                )
            )
        elif isinstance(
            module,
            (torch.nn.ReLU, torch.nn.ELU, torch.nn.SiLU, torch.nn.LayerNorm, torch.nn.BatchNorm1d),
        ):
            output_names = [f"z{module_index:02d}_{idx}" for idx in range(len(source_names))]
            lines.extend(
                _activation_formula_lines(name, module, source_names, output_names, top_k)
            )
        else:
            continue
        source_names = output_names
        if is_final:
            final_output_names = output_names

    if final_output_names:
        lines.extend(
            _final_prediction_lines(
                final_output_names,
                safe_target_names,
                task_type,
                target_scaler,
            )
        )
    else:
        lines.append("  # No output head was found; formula is incomplete.")

    lines.extend(_formula_fidelity_lines(top_k, coverage_values))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _formula_metric_lines(
    prefix: str,
    metrics: dict[str, float],
    task_type: str,
    target_names: list[str] | None = None,
) -> list[str]:
    if task_type == "classification":
        names = ["accuracy", "balanced_accuracy", "f1", "rocauc"]
    else:
        names = ["mae", "rmse", "r2"]
    lines = []
    for name in names:
        value = metrics.get(name, float("nan"))
        if np.isfinite(value):
            lines.append(f"  {prefix}_{name} = {value:.8g}")
        else:
            lines.append(f"  {prefix}_{name} = nan")
    for target_name in target_names or []:
        suffix = _target_metric_suffix(target_name)
        for name in names:
            key = f"{name}{suffix}"
            if key not in metrics:
                continue
            value = metrics.get(key, float("nan"))
            label = f"{prefix}_{name}_{_safe_symbol(target_name)}"
            if np.isfinite(value):
                lines.append(f"  {label} = {value:.8g}")
            else:
                lines.append(f"  {label} = nan")
    return lines


def _formula_fidelity_lines(top_k: int, coverage_values: list[float]) -> list[str]:
    lines = ["", "Formula fidelity"]
    if top_k <= 0:
        lines.append("  exactness = exact_untruncated_after_pruning")
        lines.append("  term_display = all nonzero terms are shown")
        return lines
    finite = [value for value in coverage_values if np.isfinite(value)]
    if finite:
        lines.append("  exactness = readable_truncated_after_pruning")
        lines.append(f"  mean_abs_coefficient_coverage = {float(np.mean(finite)):.8g}")
        lines.append(f"  min_abs_coefficient_coverage = {float(np.min(finite)):.8g}")
        lines.append(f"  equations_counted = {len(finite)}")
    else:
        lines.append("  exactness = readable_truncated_after_pruning")
        lines.append("  mean_abs_coefficient_coverage = nan")
        lines.append("  min_abs_coefficient_coverage = nan")
    lines.append("  use --formula-top-k 0 to export the exact full formula.")
    return lines


def _final_prediction_lines(
    raw_names: list[str],
    safe_target_names: list[str],
    task_type: str,
    target_scaler: TargetScaler,
) -> list[str]:
    lines = ["", "Final prediction"]
    if task_type == "classification":
        for idx, raw_name in enumerate(raw_names):
            safe_target = safe_target_names[idx] if idx < len(safe_target_names) else f"target_{idx}"
            lines.append(f"  prob_{safe_target} = sigmoid({raw_name})")
        return lines

    means = np.asarray(target_scaler.mean).reshape(-1)
    stds = np.asarray(target_scaler.std).reshape(-1)
    for idx, raw_name in enumerate(raw_names):
        safe_target = safe_target_names[idx] if idx < len(safe_target_names) else f"target_{idx}"
        if target_scaler.mode == "standard":
            std = float(stds[idx]) if idx < len(stds) else float(stds[0])
            mean = float(means[idx]) if idx < len(means) else float(means[0])
            lines.append(
                f"  pred_{safe_target} = "
                f"{std:.8g}*{raw_name} + {mean:.8g}"
            )
        else:
            lines.append(f"  pred_{safe_target} = {raw_name}")
    return lines


def parent_name(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def source_names_for(
    name: str,
    names_by_module: dict[str, list[str] | None],
    input_names: list[str] | None,
) -> list[str] | None:
    if name.startswith("output_heads."):
        prop_key = name.split(".", 1)[1]
        group_key = prop_key.split("_p", 1)[0]
        return (
            names_by_module.get(f"target_blocks.{prop_key}")
            or names_by_module.get(f"property_blocks.{prop_key}")
            or names_by_module.get(f"group_blocks.{group_key}")
            or names_by_module.get("common_block")
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
        group_key = prop_key.split("_p", 1)[0]
        return (
            names_by_module.get(parent_name(name))
            or names_by_module.get(f"property_blocks.{prop_key}")
            or names_by_module.get(f"group_blocks.{group_key}")
        )
    return names_by_module.get(parent_name(name))


def remember_output_names(
    name: str,
    output_names: list[str],
    names_by_module: dict[str, list[str] | None],
) -> None:
    names_by_module[name] = output_names
    parent = parent_name(name)
    if parent:
        names_by_module[parent] = output_names
    for prefix in ("common_block", "group_blocks", "property_blocks", "target_blocks"):
        if name.startswith(prefix + "."):
            parts = name.split(".")
            if prefix == "common_block":
                names_by_module["common_block"] = output_names
            elif len(parts) >= 2:
                names_by_module[".".join(parts[:2])] = output_names


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
        .replace("[", "")
        .replace("]", "")
        .replace(":", "_")
        .replace(";", "_")
    )


def _selected_terms(
    coefficients: list[tuple[float, str]],
    top_k: int,
    min_abs: float,
) -> list[tuple[float, str]]:
    filtered = [(coef, text) for coef, text in coefficients if abs(coef) > min_abs]
    filtered.sort(key=lambda item: abs(item[0]), reverse=True)
    return filtered if top_k <= 0 else filtered[:top_k]


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _format_terms(terms: list[tuple[float, str]]) -> str:
    if not terms:
        return "0"
    chunks = []
    for idx, (coef, text) in enumerate(terms):
        magnitude = _format_number(abs(coef))
        if idx == 0:
            prefix = "- " if coef < 0 else ""
        else:
            prefix = " - " if coef < 0 else " + "
        chunks.append(f"{prefix}{magnitude}*{text}")
    return "".join(chunks)


def _format_vector(values: list[float], top_k: int) -> str:
    max_items = 0 if top_k <= 0 else 16
    if max_items <= 0 or len(values) <= max_items:
        return "[" + ", ".join(_format_number(float(value)) for value in values) + "]"
    head = values[: max_items // 2]
    tail = values[-(max_items // 2) :]
    return (
        "["
        + ", ".join(_format_number(float(value)) for value in head)
        + f", ... ({len(values)} values) ..., "
        + ", ".join(_format_number(float(value)) for value in tail)
        + "]"
    )


def _compact_names(names: list[str] | None) -> str:
    if not names:
        return "[]"
    if len(names) <= 12:
        return "[" + ", ".join(names) + "]"
    head = ", ".join(names[:6])
    tail = ", ".join(names[-3:])
    return f"[{head}, ... ({len(names)} values) ..., {tail}]"


def _append_truncation_note(
    lines: list[str],
    selected: list[tuple[float, str]],
    all_terms: list[tuple[float, str]],
    top_k: int,
    min_abs: float,
    coverage_values: list[float],
) -> None:
    all_nonzero = _selected_terms(all_terms, 0, min_abs)
    total_abs = sum(abs(coef) for coef, _ in all_nonzero)
    selected_abs = sum(abs(coef) for coef, _ in selected)
    coverage = 1.0 if total_abs <= 0 else min(1.0, selected_abs / total_abs)
    coverage_values.append(float(coverage))
    total = len(all_nonzero)
    if total > len(selected):
        lines.append(
            f"    # shown {len(selected)} of {total} nonzero terms; "
            f"abs_coef_coverage={coverage:.4f}"
        )


def _fastkan_formula_lines(
    name: str,
    module: FastKANLinear,
    source_names: list[str] | None,
    output_names: list[str],
    module_index: int,
    top_k: int,
    min_abs: float,
    coverage_values: list[float],
) -> list[str]:
    lines = [f"  # {name}: FastKANLinear({module.in_features}->{module.out_features})"]
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
    lines.append(f"  # nonzero_terms={nonzero}, grid={grid}, h={denominator:.6g}")

    ln_prefix = None
    if module.layernorm is not None:
        ln_prefix = f"ln{module_index:02d}"
        gamma = [float(value) for value in module.layernorm.weight.detach().cpu().tolist()]
        beta = [float(value) for value in module.layernorm.bias.detach().cpu().tolist()]
        lines.append(
            f"  # {ln_prefix}_i = LayerNorm_i({_compact_names(source_names)}, "
            f"gamma={_format_vector(gamma, top_k)}, beta={_format_vector(beta, top_k)}, "
            f"eps={module.layernorm.eps:g})"
        )

    for out_idx, output_name in enumerate(output_names):
        terms: list[tuple[float, str]] = []
        if base_weight is not None:
            for in_idx in range(module.in_features):
                coef = float(base_weight[out_idx, in_idx])
                terms.append((coef, f"silu({_source_name(source_names, in_idx)})"))
            if base_bias is not None:
                terms.append((float(base_bias[out_idx]), "1"))
        for in_idx in range(module.in_features):
            source = _source_name(source_names, in_idx)
            spline_source = f"{ln_prefix}_{in_idx}" if ln_prefix is not None else source
            for grid_idx, center in enumerate(grid):
                flat_idx = in_idx * len(grid) + grid_idx
                coef = float(spline_weight[out_idx, flat_idx])
                terms.append((coef, f"rbf({spline_source}; c={center:.6g}, h={denominator:.6g})"))
        selected = _selected_terms(terms, top_k, min_abs)
        lines.append(f"  {output_name} = {_format_terms(selected)}")
        _append_truncation_note(lines, selected, terms, top_k, min_abs, coverage_values)
    lines.append("")
    return lines


def _spline_formula_lines(
    name: str,
    module: KANLinear,
    source_names: list[str] | None,
    output_names: list[str],
    top_k: int,
    min_abs: float,
    coverage_values: list[float],
) -> list[str]:
    lines = [f"  # {name}: KANLinear({module.in_features}->{module.out_features})"]
    base_weight = module.base_weight.detach().cpu()
    spline_weight = module.scaled_spline_weight.detach().cpu()
    knots = [float(value) for value in module.grid[0].detach().cpu().tolist()]
    nonzero = int(torch.count_nonzero(base_weight).item() + torch.count_nonzero(spline_weight).item())
    lines.append(
        f"  # nonzero_terms={nonzero}, spline_order={module.spline_order}, knots={knots}"
    )
    for out_idx, output_name in enumerate(output_names):
        terms: list[tuple[float, str]] = []
        for in_idx in range(module.in_features):
            source = _source_name(source_names, in_idx)
            terms.append((float(base_weight[out_idx, in_idx]), f"silu({source})"))
            for basis_idx in range(spline_weight.shape[-1]):
                coef = float(spline_weight[out_idx, in_idx, basis_idx])
                terms.append((coef, f"spline_b{in_idx}_{basis_idx}({source})"))
        selected = _selected_terms(terms, top_k, min_abs)
        lines.append(f"  {output_name} = {_format_terms(selected)}")
        _append_truncation_note(lines, selected, terms, top_k, min_abs, coverage_values)
    lines.append("")
    return lines


def _linear_formula_lines(
    name: str,
    module: torch.nn.Linear,
    source_names: list[str] | None,
    output_names: list[str],
    top_k: int,
    min_abs: float,
    coverage_values: list[float],
) -> list[str]:
    lines = [f"  # {name}: Linear({module.in_features}->{module.out_features})"]
    weight = module.weight.detach().cpu()
    bias = module.bias.detach().cpu() if module.bias is not None else None
    nonzero = int(torch.count_nonzero(weight).item())
    if bias is not None:
        nonzero += int(torch.count_nonzero(bias).item())
    lines.append(f"  # nonzero_terms={nonzero}")
    for out_idx, output_name in enumerate(output_names):
        terms = [
            (float(weight[out_idx, in_idx]), _source_name(source_names, in_idx))
            for in_idx in range(module.in_features)
        ]
        if bias is not None:
            terms.append((float(bias[out_idx]), "1"))
        selected = _selected_terms(terms, top_k, min_abs)
        lines.append(f"  {output_name} = {_format_terms(selected)}")
        _append_truncation_note(lines, selected, terms, top_k, min_abs, coverage_values)
    lines.append("")
    return lines


def write_matbench_records(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    predictions_by_model: dict[str, dict[int, list[Any]]],
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

    if args.dataset == "matbench_elastic":
        return write_elastic_matbench_records(args, rows, predictions_by_model, output_dir)

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


def write_elastic_matbench_records(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    predictions_by_model: dict[str, dict[int, list[Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    from matbench.task import MatbenchTask
    from monty.json import MontyEncoder

    source_tasks = ["matbench_log_gvrh", "matbench_log_kvrh"]
    records = {}
    for model_name, fold_predictions in predictions_by_model.items():
        model_records = {}
        for target_idx, task_name in enumerate(source_tasks):
            task = MatbenchTask(task_name, autoload=False)
            task.load()
            for fold, predictions in sorted(fold_predictions.items()):
                pred_array = np.asarray(predictions, dtype=float)
                if pred_array.ndim == 1:
                    if len(source_tasks) > 1:
                        raise ValueError("Elastic predictions must be 2D with one column per target")
                    target_predictions = pred_array.tolist()
                else:
                    target_predictions = pred_array[:, target_idx].tolist()
                row = next(
                    item
                    for item in rows
                    if str(item["model"]) == model_name and int(item["fold"]) == int(fold)
                )
                task.record(
                    fold,
                    target_predictions,
                    params={
                        key: _json_scalar(value)
                        for key, value in row.items()
                        if not key.startswith("test_") and not key.startswith("best_val_")
                    },
                )

            record_path = output_dir / f"matbench-record-{task_name}-{model_name}.json"
            record_path.write_text(
                json.dumps(task.as_dict(), indent=2, cls=MontyEncoder),
                encoding="utf-8",
            )
            model_records[task_name] = {
                "record_path": str(record_path),
                "scores": _serialize_matbench_scores(task),
            }
        records[model_name] = model_records
    return records


def _serialize_matbench_scores(task) -> Any:
    try:
        scores = task.scores
        # Matbench 0.1.x returns RecursiveDotDict here.  Its __getattr__ creates
        # missing dictionary entries, so hasattr(scores, "as_dict") is always
        # true even though as_dict is not a callable method.  Recursively copy
        # the mapping instead of probing attributes.
        return _json_value(scores)
    except Exception as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
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
        headers.extend(["best_val_mae", "best_val_rmse", "best_val_r2", "test_mae", "test_rmse", "test_r2"])
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
        "inner_fold",
        "inner_n_splits",
        "task_type",
        "target",
        "target_names",
        "n_targets",
        "model",
        "architecture",
        "block_type",
        "activation",
        "kan_impl",
        "prune_kan_fraction",
        "prune_mode",
        "pruned_edges",
        "total_kan_edges",
        "prune_finetune_epochs",
        "prune_finetune_seconds",
        "kan_l1_lambda",
        "kan_sparsity_lambda",
        "kan_sparsity_mode",
        "params_before_prune",
        "params_after_prune",
        "params_pruned",
        "params_pruned_pct",
        "formula_path",
        "spline_exact_formula_test_mae",
        "spline_symbolic_formula_path",
        "spline_symbolic_json_path",
        "spline_symbolic_test_target_mae",
        "spline_symbolic_test_teacher_mae",
        "spline_symbolic_test_fidelity_r2_pct",
        "spline_symbolic_active_features",
        "spline_symbolic_operators",
        "spline_symbolic_n_edges",
        "symbolic_kan_formula_path",
        "symbolic_kan_json_path",
        "symbolic_kan_soft_test_mae",
        "symbolic_kan_hard_test_mae",
        "symbolic_kan_hard_test_rmse",
        "symbolic_kan_hard_test_r2",
        "symbolic_kan_test_teacher_mae",
        "symbolic_kan_test_fidelity_r2_pct",
        "symbolic_kan_active_features",
        "symbolic_kan_operators",
        "symbolic_kan_active_units",
        "symbolic_kan_hardening_epochs",
        "symbolic_kan_hardening_seconds",
        "simple_formula_path",
        "simple_formula_json_path",
        "simple_formula_n_inputs",
        "simple_formula_n_searched_inputs",
        "simple_formula_n_terms",
        "simple_formula_method",
        "simple_formula_functions",
        "simple_formula_inputs",
        "simple_formula_requested_coverage_pct",
        "simple_formula_conformal_radius",
        "simple_formula_test_empirical_coverage_pct",
        "simple_formula_test_teacher_mae",
        "simple_formula_test_target_mae",
        "simple_formula_test_fidelity_r2_pct",
        "simple_formula_calibration_size",
        "params",
        "effective_params",
        "pruned_params",
        "featurizer_preset",
        "n_features",
        "kan_grid_size",
        "kan_spline_order",
        "common_dims",
        "group_dims",
        "property_dims",
        "target_dims",
        "symbolic_hidden_dims",
        "symbolic_edges_per_unit",
        "symbolic_primitives",
        "symbolic_temperature_start",
        "symbolic_temperature_end",
        "symbolic_projection_top_k",
        "lr",
        "weight_decay",
        "loss",
        "max_epochs",
        "epochs_ran",
        "best_epoch",
        "early_stopping_monitor",
        "early_stopping_min_delta",
        "early_stopping_patience",
        "restore_best_state",
        "best_val_mae",
        "best_val_rmse",
        "best_val_r2",
        "best_val_rocauc",
        "best_val_accuracy",
        "best_val_balanced_accuracy",
        "best_val_f1",
        "test_mae",
        "test_rmse",
        "test_r2",
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
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be non-negative")
    if not 0.0 <= args.prune_kan_fraction < 1.0:
        raise ValueError("--prune-kan-fraction must be in [0, 1)")
    if args.prune_finetune_epochs < 0:
        raise ValueError("--prune-finetune-epochs must be non-negative")
    if args.kan_l1_lambda < 0:
        raise ValueError("--kan-l1-lambda must be non-negative")
    if args.inner_fold_index is not None and args.val_ratio == 0:
        raise ValueError("--inner-fold-index requires --val-ratio > 0")
    if args.distill_simple_formula and args.skip_test_eval:
        raise ValueError("Simple formula distillation requires outer test evaluation")
    if args.simple_formula_max_terms < 1:
        raise ValueError("--simple-formula-max-terms must be positive")
    if args.simple_formula_epsilon <= 0 or args.simple_formula_exp_clip <= 0:
        raise ValueError("symbolic epsilon and exp clip must be positive")
    if args.simple_formula_method == "symbolic" and not any(
        name not in {"product", "ratio"} for name in args.simple_formula_functions
    ):
        raise ValueError("symbolic regression needs at least one unary function")
    if not args.symbolic_hidden_dims or any(
        value < 1 for value in args.symbolic_hidden_dims
    ):
        raise ValueError("--symbolic-hidden-dims must contain positive widths")
    if args.symbolic_edges_per_unit < 1:
        raise ValueError("--symbolic-edges-per-unit must be positive")
    if (
        args.symbolic_temperature_start <= 0
        or args.symbolic_temperature_end <= 0
    ):
        raise ValueError("Symbolic-KAN temperatures must be positive")
    if not 0 < args.symbolic_target_density <= 1:
        raise ValueError("--symbolic-target-density must be in (0, 1]")
    if not 0 <= args.symbolic_unit_threshold <= 1:
        raise ValueError("--symbolic-unit-threshold must be in [0, 1]")
    if args.symbolic_projection_top_k < 1:
        raise ValueError("--symbolic-projection-top-k must be positive")
    if args.symbolic_hardening_epochs < 0:
        raise ValueError("--symbolic-hardening-epochs must be non-negative")
    if args.symbolify_spline_kan and "direct-spline" not in args.models:
        raise ValueError("--symbolify-spline-kan requires --models direct-spline")
    if args.spline_symbolic_grid_size < 3:
        raise ValueError("--spline-symbolic-grid-size must be at least 3")
    if args.spline_symbolic_iterations < 1:
        raise ValueError("--spline-symbolic-iterations must be positive")
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
    json_temp = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    json_temp.replace(json_path)
    csv_temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ordered_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    csv_temp.replace(csv_path)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
