from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed five-fold KAN results with completed official MODNet folds."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--official-fold-csv", required=True)
    parser.add_argument("--kan-output-dir", required=True)
    parser.add_argument("--kan-fold-csv", default=None)
    parser.add_argument("--model", default="fastkan")
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=0.02,
        help="Required mean relative MAE reduction for regression.",
    )
    parser.add_argument(
        "--min-absolute-improvement",
        type=float,
        default=0.0,
        help="Required mean absolute ROC-AUC increase for classification.",
    )
    parser.add_argument("--min-fold-wins", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def latest_kan_csv(directory: Path, dataset: str) -> Path:
    matches = list(directory.glob(f"modnet-kan-{dataset}-*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No modnet-kan-{dataset}-*.csv found under {directory}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def rows_by_fold(
    rows: list[dict[str, str]],
    metric: str,
    source: str,
    model: str | None = None,
) -> dict[int, dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        raw_fold = row.get("fold")
        if raw_fold is None or raw_fold == "":
            continue
        fold = int(raw_fold)
        finite_float(row.get(metric), f"{source} fold {fold} {metric}")
        if fold in selected:
            raise ValueError(f"Duplicate {source} row for fold {fold}")
        selected[fold] = row
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.expected_folds < 1:
        raise ValueError("--expected-folds must be positive")
    if not 0 <= args.min_fold_wins <= args.expected_folds:
        raise ValueError("--min-fold-wins must be between 0 and --expected-folds")

    official_path = Path(args.official_fold_csv)
    kan_output_dir = Path(args.kan_output_dir)
    kan_path = (
        Path(args.kan_fold_csv)
        if args.kan_fold_csv
        else latest_kan_csv(kan_output_dir, args.dataset)
    )
    official_rows = read_csv(official_path)
    kan_rows = read_csv(kan_path)
    if not official_rows or not kan_rows:
        raise ValueError("Official and KAN fold CSV files must both contain rows")

    task_type = str(
        next(
            (
                row.get("task_type")
                for row in kan_rows
                if row.get("model") == args.model and row.get("task_type")
            ),
            official_rows[0].get("task_type", "regression"),
        )
    )
    classification = task_type == "classification"
    official_metric = "rocauc" if classification else "mae"
    kan_metric = "test_rocauc" if classification else "test_mae"
    official_by_fold = rows_by_fold(
        official_rows, official_metric, "official MODNet"
    )
    kan_by_fold = rows_by_fold(
        kan_rows, kan_metric, "KAN", model=args.model
    )
    expected = set(range(args.expected_folds))
    if set(official_by_fold) != expected:
        raise ValueError(
            f"Official folds are {sorted(official_by_fold)}, expected {sorted(expected)}"
        )
    if set(kan_by_fold) != expected:
        raise ValueError(
            f"KAN folds are {sorted(kan_by_fold)}, expected {sorted(expected)}"
        )

    fold_rows: list[dict[str, Any]] = []
    improvements = []
    wins = 0
    for fold in sorted(expected):
        official_value = finite_float(
            official_by_fold[fold][official_metric],
            f"official fold {fold} {official_metric}",
        )
        kan_value = finite_float(
            kan_by_fold[fold][kan_metric],
            f"KAN fold {fold} {kan_metric}",
        )
        if classification:
            improvement = kan_value - official_value
            better = kan_value > official_value
        else:
            if official_value == 0:
                raise ValueError(f"Official MAE is zero for fold {fold}")
            improvement = (official_value - kan_value) / abs(official_value)
            better = kan_value < official_value
        improvements.append(improvement)
        wins += int(better)
        fold_rows.append(
            {
                "dataset": args.dataset,
                "fold": fold,
                "task_type": task_type,
                "official_metric": official_metric,
                "official_value": official_value,
                "kan_model": args.model,
                "kan_metric": kan_metric,
                "kan_value": kan_value,
                "improvement": improvement,
                "improvement_pct": 100.0 * improvement,
                "kan_better": better,
            }
        )

    official_mean = mean(row["official_value"] for row in fold_rows)
    kan_mean = mean(row["kan_value"] for row in fold_rows)
    mean_improvement = mean(improvements)
    required_improvement = (
        args.min_absolute_improvement
        if classification
        else args.min_relative_improvement
    )
    passes = mean_improvement >= required_improvement and wins >= args.min_fold_wins
    params = [
        finite_float(row.get("params_after_prune"), "KAN params_after_prune")
        for row in kan_rows
        if row.get("model") == args.model and row.get("params_after_prune")
    ]
    summary = {
        "dataset": args.dataset,
        "task_type": task_type,
        "official_fold_csv": str(official_path),
        "kan_fold_csv": str(kan_path),
        "kan_model": args.model,
        "folds": args.expected_folds,
        "official_metric": official_metric,
        "official_mean": official_mean,
        "kan_metric": kan_metric,
        "kan_mean": kan_mean,
        "mean_improvement": mean_improvement,
        "mean_improvement_pct": 100.0 * mean_improvement,
        "required_improvement": required_improvement,
        "required_improvement_pct": 100.0 * required_improvement,
        "kan_fold_wins": wins,
        "required_fold_wins": args.min_fold_wins,
        "kan_params_mean": mean(params) if params else None,
        "passes_fixed5fold_gate": passes,
        "comparison_note": (
            "Completed official MODNet result versus one fixed, single-model KAN; "
            "no outer-fold hyperparameter selection."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"fixed5fold-comparison-{args.dataset}.csv"
    json_path = output_dir / f"fixed5fold-comparison-{args.dataset}.json"
    write_csv(csv_path, fold_rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    status = "PASS" if passes else "FAIL"
    unit = "ROC-AUC points" if classification else "relative MAE"
    print(
        f"{args.dataset}: {status} | official={official_mean:.6g} | "
        f"{args.model}={kan_mean:.6g} | mean improvement={mean_improvement:.3%} "
        f"{unit} | fold wins={wins}/{args.expected_folds}",
        flush=True,
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
