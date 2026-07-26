from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from cgcnn_pyg_kan.feature_patterns import analyze_feature_patterns


SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_kan_feature_patterns.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_kan_feature_patterns", SUMMARY_PATH
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark_modnet_kan.py"
)
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "benchmark_modnet_kan_feature_patterns", BENCHMARK_PATH
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)


def test_feature_pattern_analysis_finds_dominant_linear_feature() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(128, 3)).astype(np.float32)
    targets = (3.0 * features[:, 0] + 0.1 * features[:, 1]).reshape(-1, 1)
    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 0.1, 0.0]]))

    payload = analyze_feature_patterns(
        model,
        features,
        targets,
        ["dominant", "weak", "unused"],
        ["target"],
        torch.device("cpu"),
        "regression",
        0.0,
        1.0,
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        candidate_count=3,
        top_k=2,
        ale_bins=8,
        permutation_repeats=2,
        max_samples=128,
        seed=11,
    )

    assert payload["features"][0]["feature"] == "dominant"
    assert payload["features"][0]["permutation_importance"] > 1.0
    assert payload["features"][0]["ale"]["target"]["shape"] == "increasing"
    assert payload["features"][-1]["permutation_importance"] <= 1e-8


def test_benchmark_exports_feature_pattern_artifact(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    raw_features = rng.normal(size=(64, 2)).astype(np.float32)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler(feature_range=(-0.5, 0.5))),
        ]
    )
    scaled_features = pipeline.fit_transform(raw_features).astype(np.float32)
    targets = (2.0 * scaled_features[:, 0]).reshape(-1, 1)
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 0.0]]))
    prepared = {
        "x_test": scaled_features,
        "y_test": targets,
        "selected_features": ["signal", "noise"],
        "target_scaler": BENCHMARK.TargetScaler(targets, mode="none"),
        "feature_pipeline": pipeline,
        "processor": None,
    }
    args = SimpleNamespace(
        batch_size=32,
        feature_pattern_candidates=2,
        feature_pattern_top_k=1,
        feature_pattern_ale_bins=6,
        feature_pattern_permutation_repeats=1,
        feature_pattern_max_samples=64,
        seed=7,
        output_dir=str(tmp_path),
        dataset="example",
    )

    row = BENCHMARK.export_feature_patterns(
        model,
        prepared,
        args,
        torch.device("cpu"),
        "regression",
        ["target"],
        "fastkan",
        0,
    )

    path = Path(row["feature_pattern_path"])
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["evaluation_split"] == "outer_test"
    assert payload["features"][0]["feature"] == "signal"


def _pattern_payload(model: str, fold: int) -> dict[str, object]:
    return {
        "dataset": "example",
        "fold": fold,
        "model": model,
        "target_names": ["y"],
        "features": [
            {
                "feature": "shared_feature",
                "permutation_importance": 2.0 + 0.1 * fold,
                "selected_for_ale": True,
                "ale": {
                    "y": {
                        "shape": "u_shaped",
                        "points": [
                            {"quantile": 0.10, "raw_value": -1.0, "effect": 1.0},
                            {"quantile": 0.30, "raw_value": -0.5, "effect": 0.25},
                            {"quantile": 0.50, "raw_value": 0.0, "effect": 0.0},
                            {"quantile": 0.70, "raw_value": 0.5, "effect": 0.25},
                            {"quantile": 0.90, "raw_value": 1.0, "effect": 1.0},
                        ],
                    }
                },
            }
        ],
    }


def test_cross_model_summary_requires_performance_and_curve_consensus(
    tmp_path: Path,
) -> None:
    for model in ("fastkan", "spline"):
        for fold in range(5):
            path = tmp_path / f"feature-patterns-example-fold{fold}-{model}.json"
            path.write_text(
                json.dumps(_pattern_payload(model, fold)),
                encoding="utf-8",
            )
        comparison = {
            "kan_model": model,
            "passes_teacher_closeness_gate": True,
            "teacher_mean": 1.02,
            "official_mean": 1.0,
            "teacher_mean_degradation_pct": 2.0,
            "teacher_fold_wins": 2,
        }
        (tmp_path / f"{model}-comparison.json").write_text(
            json.dumps(comparison),
            encoding="utf-8",
        )

    payload = SUMMARY.summarize(
        "example",
        tmp_path,
        ["fastkan", "spline"],
        [
            tmp_path / "fastkan-comparison.json",
            tmp_path / "spline-comparison.json",
        ],
        expected_folds=5,
        min_fold_frequency=4,
        min_curve_correlation=0.6,
        min_permutation_importance=0.0,
        derived_feature_functions=["identity", "square", "tanh"],
        derived_feature_min_r2=0.8,
    )

    assert payload["all_models_performance_qualified"] is True
    assert payload["consensus_feature_count"] == 1
    row = payload["rows"][0]
    assert row["feature"] == "shared_feature"
    assert row["fastkan_fold_frequency"] == 5
    assert row["spline_fold_frequency"] == 5
    assert row["cross_model_curve_correlation"] == pytest.approx(1.0)
    assert row["novel_transform"] is True
    assert row["derived_feature_kind"] == "symbolic_unary"
    assert "kan_consensus__square" in row["derived_feature_name"]
    plot_paths = SUMMARY._write_consensus_plots(payload, tmp_path)
    assert len(plot_paths) == 1
    assert Path(plot_paths[0]).is_file()
