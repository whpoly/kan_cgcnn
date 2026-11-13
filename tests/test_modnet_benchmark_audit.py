import numpy as np
import pandas as pd
import pytest
import torch
import sys
from pathlib import Path
from types import SimpleNamespace

from cgcnn_pyg_kan.kan import FastKANLinear, make_kan_mlp
from cgcnn_pyg_kan.modnet_features import MODNetFeatureProcessor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_modnet_kan import metric_is_better
from benchmark_modnet_controlled import prepare_data


def test_fastkan_without_layernorm_is_additive_and_has_no_norm_parameters():
    # A one-layer KAN without normalization is a sum of independent edge
    # functions: changing two inputs has no mixed interaction.
    model = make_kan_mlp(3, [], 1, use_layernorm=False)
    assert all(layer.layernorm is None for layer in model.modules()
               if isinstance(layer, FastKANLinear))
    a = torch.tensor([[0.2, 0.3, 0.4]])
    b, c, d = a.clone(), a.clone(), a.clone()
    b[:, 0] += 0.7
    c[:, 1] -= 0.6
    d[:, 0] += 0.7
    d[:, 1] -= 0.6
    torch.testing.assert_close(model(d) - model(b), model(c) - model(a),
                               atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("task,start,small,big", [
    ("regression", 1.0, 0.995, 0.98),
    ("classification", 0.7, 0.705, 0.72),
])
def test_validation_patience_respects_min_delta(task, start, small, big):
    assert not metric_is_better(small, start, task, min_delta=0.01)
    assert metric_is_better(big, start, task, min_delta=0.01)
    assert metric_is_better(small, start, task)
    assert not metric_is_better(float("nan"), start, task)


def test_feature_processor_does_not_refit_on_extreme_holdout():
    rng = np.random.default_rng(4)
    train = pd.DataFrame(rng.normal(size=(40, 8)))
    train.columns = [f"x{i}" for i in range(8)]
    processor = MODNetFeatureProcessor(n_features=4).fit(train, train["x0"])
    before = processor.transform(train)
    columns = list(processor.selected_columns_)
    processor.transform(train * 1e6)
    assert processor.selected_columns_ == columns
    np.testing.assert_array_equal(before, processor.transform(train))


def test_controlled_preprocessing_is_independent_of_validation_and_test_labels():
    rng = np.random.default_rng(7)
    features = pd.DataFrame(rng.normal(size=(60, 8)), columns=list("abcdefgh"))
    test_features = pd.DataFrame(rng.normal(size=(12, 8)), columns=features.columns)
    y = rng.normal(size=(60, 1)).astype(np.float32)
    yt = rng.normal(size=(12, 1)).astype(np.float32)
    train_idx, val_idx = np.arange(45), np.arange(45, 60)
    args = SimpleNamespace(n_features=4, split_seed=17, preset="synthetic")
    first = prepare_data(features, test_features, y, yt, train_idx, val_idx, args)
    changed = y.copy()
    changed[val_idx] = 1e6
    second = prepare_data(features, test_features * 1e6, changed, yt * 1e6,
                          train_idx, val_idx, args)
    assert first["selected_features"] == second["selected_features"]
    assert first["target_scaler"].as_dict() == second["target_scaler"].as_dict()
    np.testing.assert_array_equal(first["x_train"], second["x_train"])
    np.testing.assert_array_equal(first["y_train_scaled"], second["y_train_scaled"])
