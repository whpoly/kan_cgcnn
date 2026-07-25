from __future__ import annotations

import numpy as np
import torch

from cgcnn_pyg_kan.kan import KANMLP
from cgcnn_pyg_kan.spline_symbolic import fit_edge_function
from cgcnn_pyg_kan.symbolic_kan import SymbolicKAN, export_symbolic_kan


def test_symbolic_kan_defaults_to_one_direct_symbolic_layer() -> None:
    model = SymbolicKAN(6, ["target"])

    assert model.hidden_dims == [5]
    assert len(model.networks[0].layers) == 1
    assert model.networks[0].layers[0].out_features == 5
    assert model.primitives == ("identity", "square", "tanh", "product")


def test_symbolic_kan_export_folds_input_scaling_into_raw_formula() -> None:
    model = SymbolicKAN(
        2,
        ["target"],
        hidden_dims=[1],
        edges_per_unit=1,
        primitives=["identity"],
    )
    layer = model.networks[0].layers[0]
    with torch.no_grad():
        layer.feature_logits[0].copy_(torch.tensor([10.0, -10.0]))
        layer.input_scale.fill_(1.5)
        layer.input_bias.fill_(-0.2)
        layer.amplitude.fill_(0.7)
        layer.term_bias.fill_(0.1)
        layer.output_bias.fill_(0.3)
        layer.unit_logits.fill_(10.0)
    model.harden(unit_threshold=0.5, projection_top_k=1)

    raw = np.asarray([7.0, 20.0], dtype=np.float32)
    scales = np.asarray([0.1, 0.01], dtype=np.float32)
    offsets = np.asarray([-0.5, -0.25], dtype=np.float32)
    preprocessed = raw * scales + offsets
    model_value = float(model(torch.from_numpy(preprocessed[None, :])).item())

    payload, text = export_symbolic_kan(
        model,
        ["density", "melting_temperature"],
        target_means=[5.0],
        target_stds=[2.0],
        feature_scales=scales,
        feature_offsets=offsets,
        feature_impute_values=[6.0, 18.0],
    )
    unit = payload["targets"][0]["layers"][0]["units"][0]
    raw_projection = float(
        unit["raw_input_scale"] * raw[unit["feature_index"]]
        + unit["raw_input_bias"]
    )
    raw_unit_value = (
        unit["amplitude"] * raw_projection + unit["term_bias"]
    )
    raw_formula_value = 2.0 * (0.3 + raw_unit_value) + 5.0

    assert np.isclose(raw_formula_value, 2.0 * model_value + 5.0)
    assert payload["input_space"] == "raw_descriptors"
    assert np.isclose(unit["raw_input_scale"], 0.15)
    assert np.isclose(unit["raw_input_bias"], -0.95)
    assert "raw_descriptor('density')" in text
    assert "preprocessed(" not in text
    assert "missing -> training impute value 6" in text


def test_paper_symbolic_kan_soft_hard_and_formula_export() -> None:
    torch.manual_seed(4)
    model = SymbolicKAN(
        6,
        ["target_a", "target_b"],
        hidden_dims=[4],
        edges_per_unit=1,
    )
    inputs = torch.randn(7, 6)
    soft = model(inputs)
    loss = soft.square().mean() + model.symbolic_regularization()
    loss.backward()

    assert soft.shape == (7, 2)
    assert torch.isfinite(soft).all()
    assert any(parameter.grad is not None for parameter in model.gate_parameters())

    model.harden(unit_threshold=0.5, projection_top_k=1)
    hard = model(inputs)
    payload, text = export_symbolic_kan(
        model,
        [f"feature_{index}" for index in range(6)],
    )

    assert hard.shape == (7, 2)
    assert torch.isfinite(hard).all()
    assert payload["method"] == "direct_additive_symbolic_kan"
    assert len(payload["targets"]) == 2
    assert "target = target_a" in text
    for network in model.networks:
        assert len(network.layers) == 1
        assert network.layer.hard
        assert len(network.selected_input_indices()) <= 8


def test_symbolic_kan_selects_shared_features_and_exports_products() -> None:
    model = SymbolicKAN(
        8,
        ["target"],
        hidden_dims=[2],
        edges_per_unit=1,
        primitives=["product"],
        max_product_terms=2,
    )
    layer = model.networks[0].layers[0]
    with torch.no_grad():
        layer.feature_logits.fill_(-10.0)
        layer.interaction_feature_logits.fill_(-10.0)
        layer.feature_logits[0, 0] = 10.0
        layer.interaction_feature_logits[0, 1] = 10.0
        layer.feature_logits[1, 2] = 10.0
        layer.interaction_feature_logits[1, 3] = 10.0
        layer.input_scale[:, 0].copy_(torch.tensor([5.0, 4.0]))
        layer.interaction_scale[:, 0].copy_(torch.tensor([6.0, 7.0]))
        layer.input_bias.zero_()
        layer.interaction_bias.zero_()
        layer.amplitude.fill_(1.0)
        layer.term_bias.zero_()
        layer.output_bias.zero_()
        layer.unit_logits.fill_(10.0)

    model.harden(
        unit_threshold=0.5,
        projection_top_k=1,
        global_feature_top_k=5,
    )
    inputs = torch.tensor(
        [[2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0]]
    )
    prediction = float(model(inputs).item())
    payload, text = export_symbolic_kan(
        model,
        [f"feature_{index}" for index in range(8)],
    )
    target = payload["targets"][0]

    assert np.isclose(prediction, (5.0 * 2.0) * (6.0 * 3.0) + (4.0 * 5.0) * (7.0 * 7.0))
    assert payload["global_feature_top_k"] == 5
    assert target["selected_global_feature_indices"] == [0, 1, 2, 3]
    assert target["active_feature_names"] == [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
    ]
    assert target["operators"] == ["product"]
    assert ")*(" in text


def test_spline_edge_symbolic_fit_recovers_sine() -> None:
    x = np.linspace(-1.0, 1.0, 201)
    y = 2.5 * np.sin(3.0 * x - 0.4) + 0.7
    result = fit_edge_function(
        x,
        y,
        ["identity", "square", "sin"],
        search_range=5.0,
        grid_size=31,
        iterations=3,
        complexity_weight=0.0,
        epsilon=1e-3,
    )

    assert result["function"] == "sin"
    assert float(result["r2"]) > 0.999


def test_spline_kan_can_disable_interlayer_layernorm() -> None:
    model = KANMLP(
        5,
        [4],
        1,
        grid_size=3,
        use_layernorm=False,
    )
    assert not any(
        isinstance(module, torch.nn.LayerNorm)
        for module in model.modules()
    )
