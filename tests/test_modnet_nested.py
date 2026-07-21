from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_modnet_kan import (  # noqa: E402
    _conformal_radius,
    _select_simple_formula,
    split_train_val,
)


def test_inner_cv_partitions_cover_outer_training_once() -> None:
    size = 41
    validation_sets = []
    for inner_fold in range(5):
        train, validation = split_train_val(
            size,
            val_ratio=0.1,
            seed=7,
            inner_fold_index=inner_fold,
            inner_n_splits=5,
        )
        assert set(train).isdisjoint(validation)
        assert sorted(np.concatenate([train, validation]).tolist()) == list(range(size))
        validation_sets.append(set(validation.tolist()))
    assert set().union(*validation_sets) == set(range(size))
    assert sum(len(values) for values in validation_sets) == size


def test_simple_formula_search_reports_five_to_ten_input_curve() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(160, 12))
    teacher = (
        1.2 * features[:, 0]
        - 0.8 * features[:, 1] ** 2
        + 0.4 * features[:, 2] * features[:, 3]
    )
    specification = _select_simple_formula(
        features,
        teacher,
        [f"feature_{idx}" for idx in range(features.shape[1])],
        min_inputs=5,
        max_inputs=10,
        max_terms=8,
        degree=2,
        seed=7,
    )
    assert [item["n_inputs"] for item in specification["input_fidelity_curve"]] == [
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert 5 <= len(specification["feature_indices"]) <= 10
    assert len(specification["coefficients"]) <= 8


def test_conformal_radius_uses_finite_sample_rank() -> None:
    residuals = np.arange(1, 21, dtype=float)
    assert _conformal_radius(residuals, coverage=0.9) == 19.0

