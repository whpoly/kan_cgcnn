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

from cgcnn_pyg_kan.data import SyntheticConfig, make_synthetic_crystal_dataset
from cgcnn_pyg_kan.model import CGCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CGCNN internal MLP vs KAN blocks.")
    parser.add_argument("--conv-nets", nargs="+", default=["mlp", "kan"], choices=["mlp", "kan"])
    parser.add_argument("--graphs", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--num-convs", type=int, default=3)
    parser.add_argument("--conv-kan-impl", choices=["spline", "fastkan"], default="fastkan")
    parser.add_argument("--conv-kan-hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conv-kan-grid-size", type=int, default=8)
    parser.add_argument("--conv-kan-spline-order", type=int, default=3)
    parser.add_argument("--forward-iters", type=int, default=80)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--output-dir", default="benchmarks")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset: list, seed: int) -> tuple[list, list, list]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_end = int(0.7 * len(indices))
    val_end = int(0.85 * len(indices))
    return (
        [dataset[i] for i in indices[:train_end]],
        [dataset[i] for i in indices[train_end:val_end]],
        [dataset[i] for i in indices[val_end:]],
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(
    model: CGCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        target = batch.y.view_as(prediction)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model: CGCNN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch)
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
    node_dim: int,
    edge_dim: int,
) -> dict[str, float | int | str]:
    device = torch.device(args.device)
    train_loader, val_loader, test_loader = loaders
    set_seed(args.seed)

    model = CGCNN(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_convs=args.num_convs,
        head_hidden_dims=args.head_hidden_dims,
        conv_net=conv_net,
        conv_kan_impl=args.conv_kan_impl,
        conv_kan_hidden_dim=args.conv_kan_hidden_dim,
        conv_kan_grid_size=args.conv_kan_grid_size,
        conv_kan_spline_order=args.conv_kan_spline_order,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_state = None
    best_val = float("inf")
    sync(device)
    train_start = time.perf_counter()
    for _ in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    sync(device)
    train_seconds = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
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
        "params": count_parameters(model),
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
        "params",
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
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    config = SyntheticConfig(num_graphs=args.graphs, seed=args.seed)
    dataset = make_synthetic_crystal_dataset(config)
    train_set, val_set, test_set = split_dataset(dataset, args.seed)
    loaders = (
        DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=args.batch_size),
        DataLoader(test_set, batch_size=args.batch_size),
    )

    rows = [
        run_conv_net(configured_conv_net, args, loaders, config.node_dim, config.edge_dim)
        for configured_conv_net in args.conv_nets
    ]
    print_table(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "args": vars(args),
        "synthetic_config": asdict(config),
        "results": rows,
    }
    json_path = output_dir / f"benchmark-{stamp}.json"
    csv_path = output_dir / f"benchmark-{stamp}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
