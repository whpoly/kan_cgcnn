"""Paired architecture audit; not an official MODNet leaderboard reproduction.

Uses all five official Matbench test folds. A fixed validation subset inside each
outer training fold chooses epochs only. Descriptor selection, imputation and
scaling are fitted exclusively on the training subset. Architecture settings are
fixed before observing any test score; seeds measure initialization variability.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_modnet_kan as bench
from cgcnn_pyg_kan.modnet_features import MODNetFeatureProcessor, make_feature_frame


VARIANTS = {
    "mlp": {"family": "mlp", "layernorm": True},
    "hybrid-fastkan": {"family": "hybrid-fastkan", "layernorm": True},
    "fastkan": {"family": "fastkan", "layernorm": True},
    "fastkan-no-layernorm": {"family": "fastkan", "layernorm": False},
    "spline": {"family": "spline", "layernorm": False},
}


def source_hashes():
    paths = [
        Path(__file__), ROOT / "scripts/benchmark_modnet_kan.py",
        ROOT / "cgcnn_pyg_kan/modnet.py", ROOT / "cgcnn_pyg_kan/kan.py",
        ROOT / "cgcnn_pyg_kan/modnet_features.py",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def prepare_data(train_features, test_features, y, y_test, train_idx, val_idx, args):
    processor = MODNetFeatureProcessor(
        n_features=args.n_features, scaler="minmax", random_state=args.split_seed,
    ).fit(train_features.iloc[train_idx], y[train_idx, 0])
    x = processor.transform(train_features)
    scale = bench.TargetScaler(y[train_idx], mode="standard")
    ys = scale.transform(y)
    return {
        "feature_preset": args.preset,
        "processor": processor, "feature_pipeline": None,
        "selected_features": processor.selected_columns_, "target_scaler": scale,
        "x_train": x[train_idx], "y_train_scaled": ys[train_idx],
        "x_val": x[val_idx], "y_val_scaled": ys[val_idx],
        "x_test": processor.transform(test_features),
        "y_test_scaled": scale.transform(y_test), "y_test": y_test,
        "train_size": len(train_idx), "val_size": len(val_idx), "test_size": len(y_test),
    }


def summarize(rows, out):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "fold-seed-results.csv", index=False)
    # Average seeds within a fold first. Folds and seeds are not independent
    # samples; do not turn 5x3 runs into 15 independent uncertainty estimates.
    folded = frame.groupby(["dataset", "variant", "fold"], as_index=False).agg(
        test_mae=("test_mae", "mean"), params=("params", "mean"),
        train_seconds=("train_seconds", "mean"),
        epochs_ran=("epochs_ran", "mean"),
    )
    folded.to_csv(out / "fold-results.csv", index=False)
    summary = folded.groupby(["dataset", "variant"], as_index=False).agg(
        mae_mean=("test_mae", "mean"), mae_fold_std=("test_mae", "std"),
        params=("params", "mean"), train_seconds=("train_seconds", "mean"),
        epochs_ran=("epochs_ran", "mean"), folds=("fold", "nunique"),
    )
    baseline = folded[folded.variant == "mlp"][["dataset", "fold", "test_mae"]]
    pairs = folded.merge(baseline, on=["dataset", "fold"], suffixes=("", "_mlp"))
    pairs["mae_delta_vs_mlp"] = pairs.test_mae - pairs.test_mae_mlp
    pairs.to_csv(out / "paired-fold-deltas.csv", index=False)
    for i, row in summary.iterrows():
        matched = pairs[(pairs.dataset == row.dataset) & (pairs.variant == row.variant)]
        summary.loc[i, "wins_vs_mlp"] = int((matched.mae_delta_vs_mlp < 0).sum())
        reference = summary[(summary.dataset == row.dataset) & (summary.variant == "mlp")]
        if len(reference):
            summary.loc[i, "mae_change_pct"] = (
                100 * (row.mae_mean / reference.iloc[0].mae_mean - 1)
            )
    summary.to_csv(out / "summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=[
        "matbench_steels", "matbench_phonons", "matbench_dielectric"])
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "fixed-architecture-paired-audit-v1",
        "official_modnet_reproduction": False,
        "selection": "local MI relevance / Pearson redundancy approximation",
        "feature_fit_scope": "inner training only",
        "epoch_selection": "held-out validation; restore lowest validation MAE",
        "refit_all_outer_training": False,
        "test_usage": "once per predeclared model/seed/fold; no test-driven selection",
        "args": vars(args), "source_sha256": source_hashes(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "numpy", "pandas", "scikit-learn", "matbench", "matminer"]},
        "architectures": {
            "mlp": [64, 32, 16], "hybrid-fastkan": [64, 32, 16],
            "fastkan": [16, 8, 4], "fastkan-no-layernorm": [16, 8, 4],
            "spline": [10, 6, 4],
        },
        "grid_size": 3, "spline_order": 3,
        "scaling": {"input": "minmax(-0.5,0.5)", "target": "standard"},
        "optimizer": {"name": "Adam", "lr": 0.003, "batch_size": 64},
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing run has different settings/code. Use a new output directory.")
    else:
        atomic_json(manifest_path, manifest)
    atomic_json(out / "status.json", {"state": "running",
                "started": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()})
    from matbench.task import MatbenchTask

    rows = []
    for dataset in args.datasets:
        task = MatbenchTask(dataset)
        if task.metadata.task_type != "regression":
            raise ValueError("This diagnostic currently supports regression tasks only.")
        args.preset = ("matminer-structure-lite" if task.metadata.input_type == "structure"
                       else "matminer-composition")
        cache = out / f"{dataset}-raw-features.csv.gz"
        if cache.exists():
            raw = pd.read_csv(cache, index_col=0)
        else:
            # These descriptors are sample-wise deterministic; nothing is fitted
            # on the full dataset. Feature filtering/selection happens below.
            raw = make_feature_frame(task.df[task.metadata.input_type].tolist(),
                                     preset=args.preset, n_jobs=1)
            raw.index = task.df.index
            raw.to_csv(cache, compression="gzip")
        if not raw.index.equals(task.df.index):
            raise ValueError("Feature row identifiers do not match the dataset.")
        for fold in args.folds:
            train_inputs, train_target = task.get_train_and_val_data(fold)
            test_inputs, test_target = task.get_test_data(fold, include_target=True)
            y = train_target.to_numpy(dtype=np.float32).reshape(-1, 1)
            y_test = test_target.to_numpy(dtype=np.float32).reshape(-1, 1)
            train_idx, val_idx = bench.split_train_val(
                len(y), args.val_ratio, args.split_seed + fold)
            prepared = prepare_data(raw.loc[train_inputs.index], raw.loc[test_inputs.index],
                                    y, y_test, train_idx, val_idx, args)
            split = {
                "train_ids": train_inputs.index[train_idx].tolist(),
                "validation_ids": train_inputs.index[val_idx].tolist(),
                "test_ids": test_inputs.index.tolist(),
                "selected_features": prepared["selected_features"],
                "target_scaler": prepared["target_scaler"].as_dict(),
            }
            atomic_json(out / f"{dataset}-fold{fold}-split.json", split)
            fold_data = {"metadata": {"dataset": dataset, "fold": fold,
                "task_type": "regression", "target": task.metadata.target,
                "target_names": [task.metadata.target]}}
            median_mae = float(np.abs(y_test - np.median(y[train_idx])).mean())
            for seed in args.seeds:
                for variant in args.variants:
                    result_file = out / f"{dataset}-f{fold}-s{seed}-{variant}.json"
                    if result_file.exists():
                        rows.append(json.loads(result_file.read_text(encoding="utf-8"))["row"])
                        continue
                    spec = VARIANTS[variant]
                    model_args = bench.parse_args([
                        "--dataset", dataset, "--models", spec["family"],
                        "--epochs", str(args.epochs), "--batch-size", "64",
                        "--lr", "0.003", "--target-scale", "standard",
                        "--early-stopping-patience", str(args.patience),
                        "--early-stopping-monitor", "validation",
                        "--kan-grid-size", "3", "--kan-spline-order", "3",
                        "--common-dims", "64", "--group-dims", "32",
                        "--property-dims", "16", "--seed", str(seed),
                        "--loss", "mae", "--log-every-epochs", "0",
                        "--device", args.device, "--output-dir", str(out),
                        "--no-matbench-records",
                    ])
                    widths = manifest["architectures"][variant]
                    model_args.kan_common_dims = [widths[0]]
                    model_args.kan_group_dims = [widths[1]]
                    model_args.kan_property_dims = [widths[2]]
                    model_args.kan_layernorm = spec["layernorm"]
                    log_path = result_file.with_suffix(".log")
                    print(f"RUN {dataset} fold={fold} seed={seed} {variant}", flush=True)
                    with log_path.open("w", encoding="utf-8") as log:
                        with contextlib.redirect_stdout(log):
                            row, predictions = bench.run_model(
                                spec["family"], fold_data, prepared, model_args,
                                torch.device(args.device))
                    row.update(dataset=dataset, variant=variant, seed=seed,
                               median_baseline_mae=median_mae)
                    atomic_json(result_file, {"row": row, "args": vars(model_args),
                                "test_ids": test_inputs.index.tolist(),
                                "target": y_test[:, 0].tolist(), "prediction": predictions})
                    rows.append(row)
                    summarize(rows, out)
                    atomic_json(out / "status.json", {"state": "running",
                                "completed_fits": len(rows), "last_result": result_file.name})
                    print(f"  MAE={row['test_mae']:.5g}, params={row['params']}, "
                          f"epochs={row['epochs_ran']}, seconds={row['train_seconds']:.1f}",
                          flush=True)
    summary = summarize(rows, out)
    atomic_json(out / "status.json", {"state": "complete", "completed_fits": len(rows)})
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
