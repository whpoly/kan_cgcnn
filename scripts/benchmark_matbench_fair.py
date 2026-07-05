from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_CHOICES = ["cgcnn", "kan-readout", "kan-conv", "kan-full", "kan-simple"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fair Matbench CGCNN vs KAN comparison profiles.")
    parser.add_argument("--dataset", default="matbench_phonons")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILE_CHOICES,
        default=["cgcnn", "kan-readout", "kan-conv", "kan-full"],
    )
    parser.add_argument("--include-simple-kan", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dims", type=int, nargs="+", default=[32])
    parser.add_argument("--kan-head-hidden-dims", type=int, nargs="+", default=[8])
    parser.add_argument("--num-convs", type=int, default=4)
    parser.add_argument("--conv-kan-hidden-dim", type=int, default=16)
    parser.add_argument("--conv-kan-impl", choices=["fastkan", "spline"], default="fastkan")
    parser.add_argument("--conv-kan-grid-size", type=int, default=3)
    parser.add_argument("--head-kan-impl", choices=["fastkan", "spline"], default=None)
    parser.add_argument("--head-kan-grid-size", type=int, default=None)
    parser.add_argument("--head-kan-spline-order", type=int, default=None)
    parser.add_argument("--edge-dim", type=int, default=41)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mlp-lr", type=float, default=None)
    parser.add_argument("--kan-lr", type=float, default=3e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=None)
    parser.add_argument("--kan-weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--epoch-pause-seconds", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def profile_specs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "cgcnn": {
            "comparison": "baseline",
            "conv_net": "mlp",
            "mlp_head_net": "mlp",
            "kan_head_net": "kan",
            "head_hidden_dims": args.head_hidden_dims,
            "kan_head_hidden_dims": args.kan_head_hidden_dims,
            "mlp_atom_features": "cgcnn",
            "mlp_edge_features": "gaussian",
            "kan_atom_features": "cgcnn",
            "kan_edge_features": "gaussian",
        },
        "kan-readout": {
            "comparison": "readout_only",
            "conv_net": "mlp",
            "mlp_head_net": "kan",
            "kan_head_net": "kan",
            "head_hidden_dims": args.kan_head_hidden_dims,
            "kan_head_hidden_dims": args.kan_head_hidden_dims,
            "mlp_atom_features": "cgcnn",
            "mlp_edge_features": "gaussian",
            "kan_atom_features": "cgcnn",
            "kan_edge_features": "gaussian",
        },
        "kan-conv": {
            "comparison": "message_only",
            "conv_net": "kan",
            "mlp_head_net": "mlp",
            "kan_head_net": "mlp",
            "head_hidden_dims": args.head_hidden_dims,
            "kan_head_hidden_dims": args.head_hidden_dims,
            "mlp_atom_features": "cgcnn",
            "mlp_edge_features": "gaussian",
            "kan_atom_features": "cgcnn",
            "kan_edge_features": "gaussian",
        },
        "kan-full": {
            "comparison": "full_model",
            "conv_net": "kan",
            "mlp_head_net": "mlp",
            "kan_head_net": "kan",
            "head_hidden_dims": args.head_hidden_dims,
            "kan_head_hidden_dims": args.kan_head_hidden_dims,
            "mlp_atom_features": "cgcnn",
            "mlp_edge_features": "gaussian",
            "kan_atom_features": "cgcnn",
            "kan_edge_features": "gaussian",
        },
        "kan-simple": {
            "comparison": "compact_ablation",
            "conv_net": "kan",
            "mlp_head_net": "mlp",
            "kan_head_net": "kan",
            "head_hidden_dims": args.head_hidden_dims,
            "kan_head_hidden_dims": args.kan_head_hidden_dims,
            "mlp_atom_features": "cgcnn",
            "mlp_edge_features": "gaussian",
            "kan_atom_features": "elemental",
            "kan_edge_features": "distance",
        },
    }


def selected_profiles(args: argparse.Namespace) -> list[str]:
    profiles = list(dict.fromkeys(args.profiles))
    if args.include_simple_kan and "kan-simple" not in profiles:
        profiles.append("kan-simple")
    return profiles


def profile_command(
    args: argparse.Namespace,
    profile: str,
    spec: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/benchmark_matbench_5fold.py",
        "--dataset",
        args.dataset,
        "--folds",
        *[str(fold) for fold in args.folds],
        "--conv-nets",
        spec["conv_net"],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--head-hidden-dims",
        *[str(dim) for dim in spec["head_hidden_dims"]],
        "--kan-head-hidden-dims",
        *[str(dim) for dim in spec["kan_head_hidden_dims"]],
        "--mlp-head-net",
        spec["mlp_head_net"],
        "--kan-head-net",
        spec["kan_head_net"],
        "--num-convs",
        str(args.num_convs),
        "--conv-kan-hidden-dim",
        str(args.conv_kan_hidden_dim),
        "--conv-kan-grid-size",
        str(args.conv_kan_grid_size),
        "--conv-kan-impl",
        args.conv_kan_impl,
        "--head-kan-impl",
        args.head_kan_impl or args.conv_kan_impl,
        "--head-kan-grid-size",
        str(args.head_kan_grid_size or args.conv_kan_grid_size),
        "--edge-dim",
        str(args.edge_dim),
        "--cutoff",
        str(args.cutoff),
        "--mlp-atom-features",
        spec["mlp_atom_features"],
        "--mlp-edge-features",
        spec["mlp_edge_features"],
        "--kan-atom-features",
        spec["kan_atom_features"],
        "--kan-edge-features",
        spec["kan_edge_features"],
        "--val-ratio",
        str(args.val_ratio),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--kan-lr",
        str(args.kan_lr),
        "--kan-weight-decay",
        str(args.kan_weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--log-every-steps",
        str(args.log_every_steps),
        "--epoch-pause-seconds",
        str(args.epoch_pause_seconds),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir / profile),
    ]
    if args.train_size is not None:
        cmd.extend(["--train-size", str(args.train_size)])
    if args.test_size is not None:
        cmd.extend(["--test-size", str(args.test_size)])
    if args.head_kan_spline_order is not None:
        cmd.extend(["--head-kan-spline-order", str(args.head_kan_spline_order)])
    if args.require_cuda:
        cmd.append("--require-cuda")
    if args.mlp_lr is not None:
        cmd.extend(["--mlp-lr", str(args.mlp_lr)])
    if args.mlp_weight_decay is not None:
        cmd.extend(["--mlp-weight-decay", str(args.mlp_weight_decay)])
    return cmd


def load_profile_summary(
    args: argparse.Namespace,
    profile: str,
    spec: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    profile_dir = output_dir / profile
    summary_path = profile_dir / f"summary-{args.dataset}.csv"
    with summary_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one model row in {summary_path}, found {len(rows)}")
    row = rows[0]
    row.update(
        {
            "profile": profile,
            "comparison": spec["comparison"],
            "output_dir": str(profile_dir),
        }
    )
    return row


def write_aggregate(args: argparse.Namespace, output_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output_dir / f"fair-summary-{args.dataset}.csv"
    json_path = output_dir / f"fair-summary-{args.dataset}.json"
    fieldnames = [
        "profile",
        "comparison",
        "model",
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
        "params_mean",
        "params_std",
        "folds",
        "optimizer_steps_total",
        "test_mae_mean",
        "test_mae_std",
        "test_rmse_mean",
        "test_rmse_std",
        "train_seconds_mean",
        "forward_ms_per_batch_mean",
        "output_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "folds": args.folds,
                "profiles": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print_table(rows)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def print_table(rows: list[dict[str, Any]]) -> None:
    print("profile | params | test_mae mean+/-std | test_rmse mean+/-std | train s | forward ms")
    print("--------+--------+-------------------+--------------------+---------+-----------")
    for row in rows:
        print(
            f"{row['profile']:<7} | "
            f"{float(row['params_mean']):.0f} | "
            f"{float(row['test_mae_mean']):.6g}+/-{float(row['test_mae_std']):.6g} | "
            f"{float(row['test_rmse_mean']):.6g}+/-{float(row['test_rmse_std']):.6g} | "
            f"{float(row['train_seconds_mean']):.6g} | "
            f"{float(row['forward_ms_per_batch_mean']):.6g}"
        )


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("benchmarks") / f"{args.dataset}-fair-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    specs = profile_specs(args)
    profiles = selected_profiles(args)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for profile in profiles:
        spec = specs[profile]
        cmd = profile_command(args, profile, spec, output_dir)
        print(f"\n### Profile: {profile} ({spec['comparison']})", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=ROOT)
        summary_rows.append(load_profile_summary(args, profile, spec, output_dir))

    if not args.dry_run:
        write_aggregate(args, output_dir, summary_rows)
        print(f"\nFair benchmark outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
