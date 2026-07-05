from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ATOM_FEATURE_CHOICES = ["onehot", "atomic_number", "elemental", "cgcnn"]
EDGE_FEATURE_CHOICES = ["gaussian", "distance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-fold Matbench CUDA benchmark.")
    parser.add_argument("--dataset", default="matbench_phonons")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--conv-nets", nargs="+", default=["mlp", "kan"], choices=["mlp", "kan"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dims", type=int, nargs="+", default=[32])
    parser.add_argument("--kan-head-hidden-dims", type=int, nargs="+", default=[8])
    parser.add_argument("--mlp-head-net", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--kan-head-net", choices=["mlp", "kan"], default="kan")
    parser.add_argument("--num-convs", type=int, default=4)
    parser.add_argument("--conv-kan-hidden-dim", type=int, default=16)
    parser.add_argument("--conv-kan-impl", choices=["fastkan", "spline"], default="fastkan")
    parser.add_argument("--conv-kan-grid-size", type=int, default=3)
    parser.add_argument("--head-kan-impl", choices=["fastkan", "spline"], default=None)
    parser.add_argument("--head-kan-grid-size", type=int, default=None)
    parser.add_argument("--head-kan-spline-order", type=int, default=None)
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
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mlp-lr", type=float, default=None)
    parser.add_argument("--kan-lr", type=float, default=3e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=None)
    parser.add_argument("--kan-weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--epoch-pause-seconds", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("benchmarks") / f"{args.dataset}-5fold-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        print(f"\n=== {args.dataset} fold {fold}, {args.epochs} epochs ===", flush=True)
        cmd = [
            sys.executable,
            "scripts/benchmark_matbench.py",
            "--dataset",
            args.dataset,
            "--fold",
            str(fold),
            "--conv-nets",
            *args.conv_nets,
            "--device",
            args.device,
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
            "--conv-kan-impl",
            args.conv_kan_impl,
            "--conv-kan-hidden-dim",
            str(args.conv_kan_hidden_dim),
            "--conv-kan-grid-size",
            str(args.conv_kan_grid_size),
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
            "--num-workers",
            str(args.num_workers),
            "--log-every-steps",
            str(args.log_every_steps),
            "--epoch-pause-seconds",
            str(args.epoch_pause_seconds),
            "--forward-iters",
            "10",
            "--warmup-iters",
            "2",
            "--output-dir",
            str(output_dir),
            "--pin-memory",
        ]
        if args.train_size is not None:
            cmd.extend(["--train-size", str(args.train_size)])
        if args.test_size is not None:
            cmd.extend(["--test-size", str(args.test_size)])
        if args.atom_features is not None:
            cmd.extend(["--atom-features", args.atom_features])
        if args.edge_features is not None:
            cmd.extend(["--edge-features", args.edge_features])
        if args.head_kan_impl is not None:
            cmd.extend(["--head-kan-impl", args.head_kan_impl])
        if args.head_kan_grid_size is not None:
            cmd.extend(["--head-kan-grid-size", str(args.head_kan_grid_size)])
        if args.head_kan_spline_order is not None:
            cmd.extend(["--head-kan-spline-order", str(args.head_kan_spline_order)])
        if args.require_cuda:
            cmd.append("--require-cuda")
        if args.mlp_lr is not None:
            cmd.extend(["--mlp-lr", str(args.mlp_lr)])
        if args.kan_lr is not None:
            cmd.extend(["--kan-lr", str(args.kan_lr)])
        if args.mlp_weight_decay is not None:
            cmd.extend(["--mlp-weight-decay", str(args.mlp_weight_decay)])
        if args.kan_weight_decay is not None:
            cmd.extend(["--kan-weight-decay", str(args.kan_weight_decay)])
        subprocess.run(cmd, check=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_matbench_results.py",
            "--input-dir",
            str(output_dir),
            "--dataset",
            args.dataset,
            "--expect-folds",
            str(len(args.folds)),
        ],
        check=True,
    )
    print(f"\n5-fold outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
