from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_modnet_kan import (  # noqa: E402
    _conformal_radius,
    _select_simple_formula,
    split_train_val,
)
from tune_modnet_kan import (  # noqa: E402
    annotate_against_mlp,
    best_trials_by_family,
    load_official_mlp_trial,
    make_trials,
    parse_args,
    resume_payload_matches_command,
    valid_compact_kan_trial,
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


def test_resume_does_not_mix_sparse_and_dense_models() -> None:
    dense_payload = {"args": {"kan_l1_lambda": 0.0}}
    dense_command = ["python", "benchmark.py", "--kan-l1-lambda", "0"]
    sparse_command = [
        "python",
        "benchmark.py",
        "--kan-l1-lambda",
        "0.0001",
        "--kan-sparsity-mode",
        "edge-group",
    ]
    sparse_payload = {
        "args": {
            "kan_l1_lambda": 0.0001,
            "kan_sparsity_mode": "edge-group",
        }
    }

    assert resume_payload_matches_command(dense_payload, dense_command)
    assert not resume_payload_matches_command(dense_payload, sparse_command)
    assert resume_payload_matches_command(sparse_payload, sparse_command)


def test_resume_rejects_old_epoch_protocol() -> None:
    payload = {
        "args": {
            "epochs": 80,
            "restore_best_state": False,
            "kan_l1_lambda": 0.0,
        }
    }
    command = [
        "python",
        "benchmark.py",
        "--epochs",
        "1000",
        "--no-restore-best-state",
        "--kan-l1-lambda",
        "0",
    ]

    assert not resume_payload_matches_command(payload, command)


def test_default_full_kan_search_is_compact_and_topology_flexible(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_modnet_kan.py",
            "--model-families",
            "fastkan",
            "spline",
            "--search-space",
            "random",
            "--num-random-trials",
            "12",
        ],
    )
    args = parse_args()
    trials = make_trials(args)

    assert {trial["model_family"] for trial in trials} == {"fastkan", "spline"}
    assert all(valid_compact_kan_trial(trial) for trial in trials)
    assert any(
        all(trial[key] == 0 for key in ("common_dim", "group_dim", "property_dim", "target_dim"))
        for trial in trials
    )
    assert {trial["n_features"] for trial in trials} == {16, 32, 64, 128}


def test_official_mlp_trial_reuses_fold_preset_without_search(tmp_path: Path) -> None:
    feature_root = tmp_path / "official_feature_folds"
    fold_dir = feature_root / "fold_2"
    fold_dir.mkdir(parents=True)
    preset = {
        "n_feat": 128,
        "num_neurons": [[256], [64], [16], [16]],
        "lr": 0.005,
        "batch_size": 32,
        "epochs": 1000,
        "loss": "mae",
        "act": "elu",
        "xscale": "standard",
    }
    (fold_dir / "metadata.json").write_text(
        json.dumps({"official_best_preset": preset}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        precomputed_feature_dir=str(feature_root),
        dataset="matbench_dielectric",
        allow_non_matbench_epochs=False,
    )

    trial = load_official_mlp_trial(args, outer_fold=2)

    assert trial["trial_id"] == "mlp_official_best_preset_outer2"
    assert trial["n_features"] == 128
    assert trial["common_dim"] == 256
    assert trial["batch_size"] == 32
    assert trial["epochs"] == 1000
    assert trial["scaler"] == "minmax"
    assert trial["declared_preset_scaler"] == "standard"


def test_kan_selection_enforces_strict_mlp_parameter_budget() -> None:
    summaries = [
        {
            "model_family": "mlp",
            "trial_id": "mlp",
            "best_val_mae_mean": 0.30,
            "effective_params_mean": 100,
        },
        {
            "model_family": "fastkan",
            "trial_id": "too-large",
            "best_val_mae_mean": 0.10,
            "effective_params_mean": 120,
        },
        {
            "model_family": "fastkan",
            "trial_id": "compact",
            "best_val_mae_mean": 0.20,
            "effective_params_mean": 80,
        },
    ]
    selected = best_trials_by_family(
        summaries,
        metric="best_val_mae",
        enforce_kan_budget=True,
    )

    assert selected["fastkan"]["trial_id"] == "compact"
    assert selected["fastkan"]["parameter_budget_ok"] is True


def test_final_summary_marks_smaller_and_better_goal() -> None:
    summaries = [
        {
            "model_family": "mlp",
            "evaluation_variant": "unpruned-benchmark",
            "params_after_prune_mean": 100.0,
            "test_mae_mean": 0.50,
        },
        {
            "model_family": "fastkan",
            "evaluation_variant": "unpruned-benchmark",
            "params_after_prune_mean": 60.0,
            "test_mae_mean": 0.40,
        },
    ]
    annotated = annotate_against_mlp(summaries, task_type="regression")
    fastkan = annotated[1]

    assert fastkan["parameter_reduction_vs_mlp_pct"] == 40.0
    assert np.isclose(fastkan["test_performance_delta_vs_mlp"], 0.10)
    assert fastkan["meets_smaller_and_better_goal"] is True
