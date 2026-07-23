from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine official MODNet, spline KAN, spline auto-symbolic, "
            "paper Symbolic-KAN soft, and hardened formula five-fold MAE."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--spline-comparison-csv", required=True)
    parser.add_argument("--symbolic-comparison-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["fold"]): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    spline = read_rows(Path(args.spline_comparison_csv))
    symbolic = read_rows(Path(args.symbolic_comparison_csv))
    if set(spline) != set(symbolic):
        raise ValueError("Spline and Symbolic-KAN comparison folds differ")

    fold_rows = []
    for fold in sorted(spline):
        official = float(spline[fold]["official_value"])
        other_official = float(symbolic[fold]["official_value"])
        if abs(official - other_official) > 1e-12:
            raise ValueError(f"Official baseline differs for fold {fold}")
        values = {
            "official_mlp_mae": official,
            "spline_kan_mae": float(spline[fold]["teacher_value"]),
            "spline_symbolic_formula_mae": float(
                spline[fold]["candidate_value"]
            ),
            "paper_symbolic_kan_soft_mae": float(
                symbolic[fold]["teacher_value"]
            ),
            "paper_symbolic_kan_formula_mae": float(
                symbolic[fold]["candidate_value"]
            ),
        }
        fold_rows.append(
            {
                "dataset": args.dataset,
                "fold": fold,
                **values,
                **{
                    f"{name}_gap_vs_mlp_pct": 100.0 * (value - official) / abs(official)
                    for name, value in values.items()
                    if name != "official_mlp_mae"
                },
            }
        )

    metric_names = [
        "official_mlp_mae",
        "spline_kan_mae",
        "spline_symbolic_formula_mae",
        "paper_symbolic_kan_soft_mae",
        "paper_symbolic_kan_formula_mae",
    ]
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "folds": len(fold_rows),
        "fold_results": fold_rows,
    }
    official_mean = mean(row["official_mlp_mae"] for row in fold_rows)
    for name in metric_names:
        values = [float(row[name]) for row in fold_rows]
        summary[f"{name}_mean"] = mean(values)
        summary[f"{name}_std"] = pstdev(values)
        if name != "official_mlp_mae":
            summary[f"{name}_gap_vs_mlp_pct"] = (
                100.0 * (mean(values) - official_mean) / abs(official_mean)
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"interpretable-kan-benchmark-{args.dataset}.csv"
    json_path = output_dir / f"interpretable-kan-benchmark-{args.dataset}.json"
    write_csv(csv_path, fold_rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"{args.dataset}: official={official_mean:.6g}, "
        f"spline={summary['spline_kan_mae_mean']:.6g}, "
        f"spline formula={summary['spline_symbolic_formula_mae_mean']:.6g}, "
        f"paper soft={summary['paper_symbolic_kan_soft_mae_mean']:.6g}, "
        f"paper formula={summary['paper_symbolic_kan_formula_mae_mean']:.6g}",
        flush=True,
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
