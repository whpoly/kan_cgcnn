from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = [
    "best_val_mae",
    "test_mae",
    "test_rmse",
    "train_seconds",
    "forward_ms_per_batch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Matbench fold JSON outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expect-folds", type=int, default=None)
    return parser.parse_args()


def _fold_from_payload(payload: dict) -> int:
    return int(payload["matbench"]["fold"])


def _steps_per_model(payload: dict) -> int:
    train_size = int(payload["matbench"]["train_size"])
    batch_size = int(payload["args"]["batch_size"])
    epochs = int(payload["args"]["epochs"])
    return math.ceil(train_size / batch_size) * epochs


def _summary_stats(values: list[float]) -> tuple[float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("nan"), float("nan")
    if len(finite_values) == 1:
        return finite_values[0], 0.0
    metric_mean = mean(finite_values)
    variance = sum((value - metric_mean) ** 2 for value in finite_values) / (
        len(finite_values) - 1
    )
    return metric_mean, math.sqrt(variance)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob(f"matbench-{args.dataset}-fold*.json"))
    if not paths:
        raise FileNotFoundError(f"No fold JSON files found in {input_dir}")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    folds = sorted({_fold_from_payload(payload) for payload in payloads})
    if args.expect_folds is not None and len(folds) != args.expect_folds:
        raise RuntimeError(f"Expected {args.expect_folds} folds, found {len(folds)}: {folds}")

    rows_by_model: dict[str, list[dict]] = defaultdict(list)
    for payload in payloads:
        fold = _fold_from_payload(payload)
        steps = _steps_per_model(payload)
        for result in payload["results"]:
            kan_impl = result.get("kan_impl", "none")
            model_id = result["conv_net"] if result["conv_net"] == "mlp" else f"kan_{kan_impl}"
            row = {"fold": fold, "optimizer_steps": steps, **result}
            rows_by_model[model_id].append(row)

    summary_rows = []
    for model_id, rows in sorted(rows_by_model.items()):
        summary = {
            "model": model_id,
            "conv_net": rows[0]["conv_net"],
            "kan_impl": rows[0].get("kan_impl", "none"),
            "head_net": rows[0].get("head_net", ""),
            "head_hidden_dims": rows[0].get("head_hidden_dims", ""),
            "head_kan_impl": rows[0].get("head_kan_impl", ""),
            "head_kan_grid_size": rows[0].get("head_kan_grid_size", ""),
            "atom_features": rows[0].get("atom_features", ""),
            "edge_features": rows[0].get("edge_features", ""),
            "node_input_dim": rows[0].get("node_input_dim", ""),
            "edge_input_dim": rows[0].get("edge_input_dim", ""),
            "folds": len(rows),
            "optimizer_steps_per_fold_mean": mean(row["optimizer_steps"] for row in rows),
            "optimizer_steps_total": sum(row["optimizer_steps"] for row in rows),
        }
        param_mean, param_std = _summary_stats([float(row["params"]) for row in rows])
        summary["params_mean"] = param_mean
        summary["params_std"] = param_std
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            metric_mean, metric_std = _summary_stats(values)
            summary[f"{metric}_mean"] = metric_mean
            summary[f"{metric}_std"] = metric_std
        summary_rows.append(summary)

    summary_csv = input_dir / f"summary-{args.dataset}.csv"
    summary_json = input_dir / f"summary-{args.dataset}.json"
    fieldnames = list(summary_rows[0].keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "folds": folds,
                "fold_files": [str(path) for path in paths],
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("model | folds | params | test_mae mean+/-std | test_rmse mean+/-std | total steps")
    print("------+-------+--------+-------------------+--------------------+------------")
    for row in summary_rows:
        print(
            f"{row['model']:<5} | "
            f"{row['folds']:<5} | "
            f"{row['params_mean']:.0f} | "
            f"{row['test_mae_mean']:.6g}+/-{row['test_mae_std']:.6g} | "
            f"{row['test_rmse_mean']:.6g}+/-{row['test_rmse_std']:.6g} | "
            f"{int(row['optimizer_steps_total'])}"
        )

    print(f"\nWrote {summary_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
