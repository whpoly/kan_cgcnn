from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_official_modnet_kan.py"
SPEC = importlib.util.spec_from_file_location("compare_official_modnet_kan", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_formula_comparison_reports_teacher_formula_and_stability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    official_path = tmp_path / "official.csv"
    _write_csv(
        official_path,
        [
            {"fold": fold, "task_type": "regression", "mae": 10.0}
            for fold in range(5)
        ],
    )

    kan_dir = tmp_path / "kan"
    kan_dir.mkdir()
    kan_rows = []
    for fold in range(5):
        formula_path = kan_dir / f"formula-{fold}.json"
        formula_path.write_text(
            json.dumps(
                {
                    "targets": [
                        {
                            "target": "y",
                            "expression": "1 + 2*z0",
                            "active_feature_names": ["feature_a", "feature_b"],
                            "variable_definitions": [
                                {
                                    "variable": "z0",
                                    "feature": "feature_a",
                                    "expression": "preprocessed('feature_a')",
                                }
                            ],
                            "test_target_mae": 10.2,
                            "test_teacher_mae": 0.1,
                            "test_fidelity_r2": 0.95,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        kan_rows.append(
            {
                "fold": fold,
                "task_type": "regression",
                "model": "direct-spline",
                "test_mae": 10.1,
                "simple_formula_test_target_mae": 10.2,
                "simple_formula_test_fidelity_r2_pct": 95.0,
                "simple_formula_inputs": "feature_a,feature_b",
                "simple_formula_n_terms": 2,
                "simple_formula_json_path": str(formula_path),
                "params_after_prune": 100,
            }
        )
    kan_path = kan_dir / "modnet-kan-example-20260723-000000.csv"
    _write_csv(kan_path, kan_rows)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--dataset",
            "example",
            "--official-fold-csv",
            str(official_path),
            "--kan-output-dir",
            str(kan_dir),
            "--model",
            "direct-spline",
            "--comparison-target",
            "formula",
            "--output-dir",
            str(kan_dir),
        ],
    )
    COMPARE.main()

    summary = json.loads(
        (kan_dir / "fixed5fold-comparison-example.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["official_mean"] == 10.0
    assert summary["kan_mean"] == 10.1
    assert summary["formula_mean"] == 10.2
    assert summary["passes_superiority_gate"] is False
    assert summary["passes_closeness_gate"] is True
    assert summary["passes_stability_gate"] is True
    assert summary["passes_fixed5fold_gate"] is True
    assert (kan_dir / "symbolic-formulas-5fold-example.txt").is_file()


def test_paper_symbolic_kan_comparison_uses_hard_formula_metric(
    tmp_path: Path,
    monkeypatch,
) -> None:
    official_path = tmp_path / "official.csv"
    _write_csv(
        official_path,
        [
            {"fold": fold, "task_type": "regression", "mae": 10.0}
            for fold in range(5)
        ],
    )
    kan_dir = tmp_path / "symbolic"
    kan_dir.mkdir()
    rows = []
    for fold in range(5):
        formula_path = kan_dir / f"paper-formula-{fold}.json"
        formula_path.write_text(
            json.dumps(
                {
                    "targets": [
                        {
                            "target": "y",
                            "expression": "h1_0",
                            "hidden_definitions": [
                                {"variable": "h1_0", "expression": "sin(x0)"}
                            ],
                            "active_feature_names": ["feature_a"],
                            "operators": ["sin"],
                            "variable_definitions": [
                                {
                                    "variable": "x0",
                                    "feature": "feature_a",
                                    "expression": "preprocessed('feature_a')",
                                }
                            ],
                            "test_target_mae": 10.2,
                            "test_teacher_mae": 0.1,
                            "test_fidelity_r2": 0.95,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "fold": fold,
                "task_type": "regression",
                "model": "symbolic-kan",
                "test_mae": 10.1,
                "symbolic_kan_hard_test_mae": 10.2,
                "symbolic_kan_test_fidelity_r2_pct": 95.0,
                "symbolic_kan_active_features": "feature_a",
                "symbolic_kan_active_units": 2,
                "symbolic_kan_json_path": str(formula_path),
                "params_after_prune": 100,
            }
        )
    _write_csv(kan_dir / "modnet-kan-example-20260723-010000.csv", rows)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--dataset",
            "example",
            "--official-fold-csv",
            str(official_path),
            "--kan-output-dir",
            str(kan_dir),
            "--model",
            "symbolic-kan",
            "--comparison-target",
            "symbolic-kan",
            "--output-dir",
            str(kan_dir),
        ],
    )
    COMPARE.main()
    summary = json.loads(
        (kan_dir / "fixed5fold-comparison-example.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["teacher_mean"] == 10.1
    assert summary["symbolic_kan_hard_mean"] == 10.2
    assert summary["formula_stability"]["mean_formula_teacher_fidelity_r2"] == 0.95
