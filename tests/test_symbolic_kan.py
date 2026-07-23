from __future__ import annotations

import numpy as np
import torch

from cgcnn_pyg_kan.kan import KANMLP
from cgcnn_pyg_kan.spline_symbolic import fit_edge_function
from cgcnn_pyg_kan.symbolic_kan import SymbolicKAN, export_symbolic_kan


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
