from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_interpretable_kan.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_interpretable_kan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_combines_all_four_interpretable_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spline = tmp_path / "spline.csv"
    symbolic = tmp_path / "symbolic.csv"
    _write(
        spline,
        [
            {
                "fold": fold,
                "official_value": 10.0,
                "teacher_value": 10.1,
                "candidate_value": 10.2,
            }
            for fold in range(5)
        ],
    )
    _write(
        symbolic,
        [
            {
                "fold": fold,
                "official_value": 10.0,
                "teacher_value": 10.3,
                "candidate_value": 10.4,
            }
            for fold in range(5)
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--dataset",
            "example",
            "--spline-comparison-csv",
            str(spline),
            "--symbolic-comparison-csv",
            str(symbolic),
            "--output-dir",
            str(tmp_path),
        ],
    )
    MODULE.main()
    summary = json.loads(
        (tmp_path / "interpretable-kan-benchmark-example.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["spline_kan_mae_mean"] == 10.1
    assert summary["spline_symbolic_formula_mae_mean"] == 10.2
    assert summary["paper_symbolic_kan_soft_mae_mean"] == 10.3
    assert summary["paper_symbolic_kan_formula_mae_mean"] == 10.4
