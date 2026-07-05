from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgcnn_pyg_kan.materials import StructureGraphConfig, structures_to_graphs
from cgcnn_pyg_kan.model import CGCNN

ATOM_FEATURE_CHOICES = ["onehot", "atomic_number", "elemental", "cgcnn"]
EDGE_FEATURE_CHOICES = ["gaussian", "distance"]


class TargetScaler:
    def __init__(self, values: torch.Tensor) -> None:
        self.mean = values.mean()
        self.std = values.std(unbiased=False).clamp_min(1e-8)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values.device)) / self.std.to(values.device)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values.device) + self.mean.to(values.device)

    def as_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CGCNN internal MLP vs KAN on Matbench structure datasets."
    )
    parser.add_argument("--dataset", default="matbench_phonons")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--conv-nets", nargs="+", default=["mlp", "kan"], choices=["mlp", "kan"])
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dims", type=int, nargs="+", default=[32])
    parser.add_argument("--kan-head-hidden-dims", type=int, nargs="+", default=[8])
    parser.add_argument("--mlp-head-net", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--kan-head-net", choices=["mlp", "kan"], default="kan")
    parser.add_argument("--num-convs", type=int, default=4)
    parser.add_argument("--conv-kan-impl", choices=["spline", "fastkan"], default="fastkan")
    parser.add_argument("--conv-kan-hidden-dim", type=int, default=16)
    parser.add_argument("--conv-kan-grid-size", type=int, default=3)
    parser.add_argument("--conv-kan-spline-order", type=int, default=3)
    parser.add_argument("--head-kan-impl", choices=["spline", "fastkan"], default=None)
    parser.add_argument("--head-kan-grid-size", type=int, default=None)
    parser.add_argument("--head-kan-spline-order", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--edge-dim", type=int, default=41)
    parser.add_argument("--max-atomic-number", type=int, default=92)
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
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mlp-lr", type=float, default=None)
    parser.add_argument("--kan-lr", type=float, default=3e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=None)
    parser.add_argument("--kan-weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--epoch-pause-seconds", type=float, default=0.0)
    parser.add_argument("--forward-iters", type=int, default=40)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--output-dir", default="benchmarks")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was set, but --device is not a CUDA device")
    return device


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


def optimizer_hparams(conv_net: str, args: argparse.Namespace) -> tuple[float, float]:
    lr_override = args.kan_lr if conv_net == "kan" else args.mlp_lr
    wd_override = args.kan_weight_decay if conv_net == "kan" else args.mlp_weight_decay
    lr = args.lr if lr_override is None else lr_override
    weight_decay = args.weight_decay if wd_override is None else wd_override
    return lr, weight_decay


def head_hparams(conv_net: str, args: argparse.Namespace) -> tuple[str, list[int]]:
    if conv_net == "kan":
        return args.kan_head_net, list(args.kan_head_hidden_dims)
    return args.mlp_head_net, list(args.head_hidden_dims)


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
            or "bn" in lower_name
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
        return list(items), list(targets)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(items), size=size, replace=False)
    return [items.iloc[i] for i in indices], [targets.iloc[i] for i in indices]


def split_train_val(graphs: list, val_ratio: float, seed: int) -> tuple[list, list]:
    if val_ratio == 0.0:
        return graphs, []
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val_ratio must be 0 or in (0, 0.5)")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(graphs), generator=generator).tolist()
    val_size = max(1, int(round(len(indices) * val_ratio)))
    val_indices = set(indices[:val_size])
    train_graphs = [graph for idx, graph in enumerate(graphs) if idx not in val_indices]
    val_graphs = [graph for idx, graph in enumerate(graphs) if idx in val_indices]
    return train_graphs, val_graphs


def apply_feature_aliases(args: argparse.Namespace) -> None:
    if args.atom_features is not None:
        args.kan_atom_features = args.atom_features
    if args.edge_features is not None:
        args.kan_edge_features = args.edge_features


def graph_config_for_conv_net(conv_net: str, args: argparse.Namespace) -> StructureGraphConfig:
    if conv_net == "mlp":
        atom_features = args.mlp_atom_features
        edge_features = args.mlp_edge_features
    elif conv_net == "kan":
        atom_features = args.kan_atom_features
        edge_features = args.kan_edge_features
    else:
        raise ValueError(f"unsupported conv_net {conv_net!r}")

    return StructureGraphConfig(
        cutoff=args.cutoff,
        edge_dim=args.edge_dim,
        max_atomic_number=args.max_atomic_number,
        atom_features=atom_features,
        edge_features=edge_features,
    )


def graph_config_payload(
    graph_config: StructureGraphConfig,
    conversion_seconds: float,
    train_size: int,
    val_size: int,
    test_size: int,
) -> dict[str, float | int | str | None]:
    return {
        **asdict(graph_config),
        "node_dim": graph_config.node_dim,
        "edge_input_dim": graph_config.edge_input_dim,
        "conversion_seconds": conversion_seconds,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }


def load_matbench_split(args: argparse.Namespace):
    from matbench.metadata import mbv01_metadata
    from matbench.task import MatbenchTask

    metadata = mbv01_metadata[args.dataset]
    if metadata.input_type != "structure":
        raise ValueError(f"{args.dataset} input_type is {metadata.input_type!r}, not 'structure'")
    if metadata.task_type != "regression":
        raise ValueError(f"{args.dataset} task_type is {metadata.task_type!r}; this script supports regression")

    task = MatbenchTask(args.dataset, autoload=False)
    task.load()
    train_structures, train_targets = task.get_train_and_val_data(args.fold)
    test_structures, test_targets = task.get_test_data(args.fold, include_target=True)
    train_structures, train_targets = select_subset(
        train_structures,
        train_targets,
        args.train_size,
        args.seed,
    )
    test_structures, test_targets = select_subset(
        test_structures,
        test_targets,
        args.test_size,
        args.seed + 1,
    )

    return {
        "metadata": {
            "dataset": args.dataset,
            "fold": args.fold,
            "target": metadata.target,
            "n_samples": metadata.n_samples,
            "train_and_val_size": len(train_structures),
            "test_size": len(test_structures),
        },
        "train_structures": train_structures,
        "train_targets": train_targets,
        "test_structures": test_structures,
        "test_targets": test_targets,
    }


def build_graphs_for_conv_net(
    args: argparse.Namespace,
    split_data: dict,
    conv_net: str,
) -> dict:
    graph_config = graph_config_for_conv_net(conv_net, args)
    start = time.perf_counter()
    train_graphs_all = structures_to_graphs(
        split_data["train_structures"],
        split_data["train_targets"],
        graph_config,
    )
    test_graphs = structures_to_graphs(
        split_data["test_structures"],
        split_data["test_targets"],
        graph_config,
    )
    conversion_seconds = time.perf_counter() - start
    train_graphs, val_graphs = split_train_val(train_graphs_all, args.val_ratio, args.seed)

    return {
        "graph_config": graph_config,
        "conversion_seconds": conversion_seconds,
        "train_graphs": train_graphs,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
    }


def train_one_epoch(
    model: CGCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: TargetScaler,
    device: torch.device,
    conv_net: str,
    epoch: int,
    epochs: int,
    log_every_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    num_batches = len(loader)
    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        target = scaler.transform(batch.y.view_as(prediction))
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * batch.num_graphs
        total_graphs += batch.num_graphs
        if log_every_steps > 0 and (step % log_every_steps == 0 or step == num_batches):
            print(
                f"[{conv_net}] epoch {epoch}/{epochs} "
                f"step {step}/{num_batches} loss={float(loss.detach()):.6g}",
                flush=True,
            )
    return total_loss / total_graphs


@torch.no_grad()
def evaluate(
    model: CGCNN,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        batch = batch.to(device)
        prediction = scaler.inverse_transform(model(batch))
        preds.append(prediction.cpu())
        targets.append(batch.y.view_as(prediction).cpu())

    pred = torch.cat(preds)
    target = torch.cat(targets)
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(F.mse_loss(pred, target)).item()
    return {"mae": mae, "rmse": rmse}


@torch.no_grad()
def benchmark_forward(
    model: CGCNN,
    batch,
    device: torch.device,
    warmup_iters: int,
    forward_iters: int,
) -> float:
    model.eval()
    batch = batch.to(device)
    for _ in range(warmup_iters):
        _ = model(batch)
    sync(device)
    start = time.perf_counter()
    for _ in range(forward_iters):
        _ = model(batch)
    sync(device)
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / forward_iters


def run_conv_net(
    conv_net: str,
    args: argparse.Namespace,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    scaler: TargetScaler,
    graph_config: StructureGraphConfig,
) -> dict[str, float | int | str]:
    device = torch.device(args.device)
    train_loader, val_loader, test_loader = loaders
    set_seed(args.seed)
    head_net, head_hidden_dims = head_hparams(conv_net, args)
    model = CGCNN(
        node_input_dim=graph_config.node_dim,
        edge_input_dim=graph_config.edge_input_dim,
        hidden_dim=args.hidden_dim,
        num_convs=args.num_convs,
        head_hidden_dims=head_hidden_dims,
        conv_net=conv_net,
        head_net=head_net,
        conv_kan_impl=args.conv_kan_impl,
        conv_kan_hidden_dim=args.conv_kan_hidden_dim,
        conv_kan_grid_size=args.conv_kan_grid_size,
        conv_kan_spline_order=args.conv_kan_spline_order,
        head_kan_impl=args.head_kan_impl,
        head_kan_grid_size=args.head_kan_grid_size,
        head_kan_spline_order=args.head_kan_spline_order,
        dropout=args.dropout,
    ).to(device)

    lr, weight_decay = optimizer_hparams(conv_net, args)
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model, weight_decay),
        lr=lr,
    )
    best_state = None
    best_val = float("nan")
    total_steps = len(train_loader) * args.epochs
    print(
        f"\n[{conv_net}] start training: epochs={args.epochs}, "
        f"steps/epoch={len(train_loader)}, total_steps={total_steps}",
        flush=True,
    )
    sync(device)
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            conv_net=conv_net,
            epoch=epoch,
            epochs=args.epochs,
            log_every_steps=args.log_every_steps,
        )
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, scaler, device)
            if not best_state or val_metrics["mae"] < best_val:
                best_val = val_metrics["mae"]
                best_state = {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                }
            print(
                f"[{conv_net}] epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.6g} val_mae={val_metrics['mae']:.6g} "
                f"best_val_mae={best_val:.6g}",
                flush=True,
            )
        else:
            print(
                f"[{conv_net}] epoch {epoch}/{args.epochs} train_loss={train_loss:.6g}",
                flush=True,
            )
        if args.epoch_pause_seconds > 0:
            time.sleep(args.epoch_pause_seconds)
    sync(device)
    train_seconds = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, scaler, device)
    first_batch = next(iter(test_loader))
    forward_ms = benchmark_forward(
        model,
        first_batch,
        device,
        warmup_iters=args.warmup_iters,
        forward_iters=args.forward_iters,
    )
    return {
        "conv_net": conv_net,
        "kan_impl": args.conv_kan_impl if conv_net == "kan" else "none",
        "head_net": head_net,
        "head_hidden_dims": " ".join(str(dim) for dim in head_hidden_dims),
        "head_kan_impl": (args.head_kan_impl or args.conv_kan_impl) if head_net == "kan" else "none",
        "head_kan_grid_size": (
            args.head_kan_grid_size or args.conv_kan_grid_size
        ) if head_net == "kan" else "none",
        "atom_features": graph_config.atom_features,
        "edge_features": graph_config.edge_features,
        "node_input_dim": graph_config.node_dim,
        "edge_input_dim": graph_config.edge_input_dim,
        "params": count_parameters(model),
        "lr": lr,
        "weight_decay": weight_decay,
        "best_val_mae": best_val,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "train_seconds": train_seconds,
        "forward_ms_per_batch": forward_ms,
    }


def print_table(rows: Iterable[dict[str, float | int | str]]) -> None:
    rows = list(rows)
    headers = [
        "conv_net",
        "kan_impl",
        "head_net",
        "head_hidden_dims",
        "head_kan_impl",
        "head_kan_grid_size",
        "atom_features",
        "edge_features",
        "node_input_dim",
        "edge_input_dim",
        "params",
        "lr",
        "weight_decay",
        "best_val_mae",
        "test_mae",
        "test_rmse",
        "train_seconds",
        "forward_ms_per_batch",
    ]
    widths = {
        header: max(len(header), *(len(_format(row[header])) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(_format(row[header]).ljust(widths[header]) for header in headers))


def _format(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    args = parse_args()
    apply_feature_aliases(args)
    set_seed(args.seed)
    device = resolve_device(args)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    split_data = load_matbench_split(args)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers

    print(f"Runtime device: {device}")
    rows = []
    graph_configs = {}
    target_scalers = {}
    matbench_metadata = dict(split_data["metadata"])
    for configured_conv_net in args.conv_nets:
        model_data = build_graphs_for_conv_net(args, split_data, configured_conv_net)
        graph_config: StructureGraphConfig = model_data["graph_config"]
        train_graphs = model_data["train_graphs"]
        val_graphs = model_data["val_graphs"]
        test_graphs = model_data["test_graphs"]
        scaler = TargetScaler(torch.cat([graph.y for graph in train_graphs]))
        loaders = (
            DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, **loader_kwargs),
            DataLoader(val_graphs, batch_size=args.batch_size, **loader_kwargs)
            if val_graphs
            else None,
            DataLoader(test_graphs, batch_size=args.batch_size, **loader_kwargs),
        )
        graph_configs[configured_conv_net] = graph_config_payload(
            graph_config,
            conversion_seconds=model_data["conversion_seconds"],
            train_size=len(train_graphs),
            val_size=len(val_graphs),
            test_size=len(test_graphs),
        )
        target_scalers[configured_conv_net] = scaler.as_dict()
        matbench_metadata["train_size"] = len(train_graphs)
        matbench_metadata["val_size"] = len(val_graphs)
        rows.append(
            run_conv_net(configured_conv_net, args, loaders, scaler, graph_config)
        )
    print_table(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "args": vars(args),
        "matbench": matbench_metadata,
        "graph_configs": graph_configs,
        "target_scalers": target_scalers,
        "runtime": runtime_info(device),
        "results": rows,
    }
    json_path = output_dir / f"matbench-{args.dataset}-fold{args.fold}-{stamp}.json"
    csv_path = output_dir / f"matbench-{args.dataset}-fold{args.fold}-{stamp}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDataset: {args.dataset}, fold {args.fold}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
