from __future__ import annotations

import numpy as np
import torch

from cgcnn_pyg_kan.kan import KANMLP
from cgcnn_pyg_kan.spline_symbolic import fit_edge_function
from cgcnn_pyg_kan.symbolic_kan import SymbolicKAN, export_symbolic_kan


def test_symbolic_kan_defaults_to_one_direct_symbolic_layer() -> None:
    model = SymbolicKAN(6, ["target"])

    assert model.hidden_dims == [4]
    assert len(model.networks[0].layers) == 1
    assert model.networks[0].layers[0].out_features == 4


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
        layer.projection_weight[0, 0].copy_(torch.tensor([2.0, -3.0]))
        layer.projection_bias[0, 0] = 0.4
        layer.gamma.fill_(1.5)
        layer.beta.fill_(-0.2)
        layer.amplitude.fill_(0.7)
        layer.output_bias.fill_(0.1)
    model.harden(unit_threshold=0.5, projection_top_k=2)

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
    indices = np.asarray(unit["projection_indices"], dtype=int)
    raw_projection = float(
        np.dot(np.asarray(unit["projection_weight"]), raw[indices])
        + unit["projection_bias"]
    )
    raw_unit_value = (
        unit["amplitude"]
        * (unit["gamma"] * raw_projection + unit["beta"])
        + unit["output_bias"]
    )
    raw_formula_value = 2.0 * raw_unit_value + 5.0

    assert np.isclose(raw_formula_value, 2.0 * model_value + 5.0)
    assert payload["input_space"] == "raw_descriptors"
    assert np.allclose(unit["projection_weight"], [0.2, -0.03])
    assert np.isclose(unit["projection_bias"], 0.15)
    assert "raw_descriptor('density')" in text
    assert "preprocessed(" not in text
    assert "missing -> training impute value 6" in text


def test_paper_symbolic_kan_soft_hard_and_formula_export() -> None:
    torch.manual_seed(4)
    model = SymbolicKAN(
        6,
        ["target_a", "target_b"],
        hidden_dims=[4, 2],
        edges_per_unit=3,
    )
    inputs = torch.randn(7, 6)
    soft = model(inputs)
    loss = soft.square().mean() + model.symbolic_regularization()
    loss.backward()

    assert soft.shape == (7, 2)
    assert torch.isfinite(soft).all()
    assert any(parameter.grad is not None for parameter in model.gate_parameters())

    model.harden(unit_threshold=0.5, projection_top_k=3)
    hard = model(inputs)
    payload, text = export_symbolic_kan(
        model,
        [f"feature_{index}" for index in range(6)],
    )

    assert hard.shape == (7, 2)
    assert torch.isfinite(hard).all()
    assert payload["method"] == "symbolic_kan_discrete_gated"
    assert len(payload["targets"]) == 2
    assert "target = target_a" in text
    for network in model.networks:
        for layer in network.layers:
            selected = layer.hard_projection_mask.sum(dim=-1)
            live = layer.hard_unit_mask.bool()
            selected_edge = layer.hard_edge_index
            assert torch.all(
                selected[live, selected_edge[live]]
                <= 3
            )


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
