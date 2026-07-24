from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


SYMBOLIC_PRIMITIVES = (
    "zero",
    "one",
    "identity",
    "square",
    "cube",
    "sin",
    "cos",
    "tanh",
    "exp",
    "log1p_abs",
    "lorentz",
    "gaussian",
    "sinh",
    "cosh",
    "product",
)


def _primitive_library(
    values: torch.Tensor,
    names: Sequence[str],
    interaction_values: torch.Tensor | None = None,
) -> torch.Tensor:
    outputs = []
    if values.shape[-1] != len(names):
        raise ValueError("Primitive parameter axis does not match primitive library")
    if interaction_values is not None and interaction_values.shape != values.shape:
        raise ValueError("Interaction primitive inputs must match primary inputs")
    for index, name in enumerate(names):
        argument = values[..., index]
        if name == "zero":
            result = torch.zeros_like(argument)
        elif name == "one":
            result = torch.ones_like(argument)
        elif name == "identity":
            result = argument
        elif name == "square":
            result = argument.square()
        elif name == "cube":
            result = argument.pow(3)
        elif name == "sin":
            result = torch.sin(argument)
        elif name == "cos":
            result = torch.cos(argument)
        elif name == "tanh":
            result = torch.tanh(argument)
        elif name == "exp":
            result = torch.exp(argument.clamp(-10.0, 10.0))
        elif name == "log1p_abs":
            result = torch.log1p(argument.abs())
        elif name == "lorentz":
            result = 1.0 / (1.0 + argument.square())
        elif name == "gaussian":
            result = torch.exp(-argument.square().clamp(0.0, 20.0))
        elif name == "sinh":
            result = torch.sinh(argument.clamp(-5.0, 5.0))
        elif name == "cosh":
            result = torch.cosh(argument.clamp(-5.0, 5.0))
        elif name == "product":
            if interaction_values is None:
                raise ValueError("Product primitive requires a second projection")
            result = argument * interaction_values[..., index]
        else:
            raise ValueError(f"Unsupported Symbolic-KAN primitive {name!r}")
        outputs.append(torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4))
    return torch.stack(outputs, dim=-1)


def _straight_through_top1(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
    return hard - scores.detach() + scores


@dataclass(frozen=True)
class SymbolicRegularization:
    selection: float = 1e-3
    entropy: float = 1.0
    nms: float = 0.1
    unit: float = 1e-3
    bias: float = 1e-4
    projection_l1: float = 1e-5
    target_density: float = 0.75


class SymbolicKANLayer(nn.Module):
    """A Symbolic-KAN layer following equations (5)--(18) of arXiv:2603.23854."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edges_per_unit: int,
        primitives: Sequence[str],
        *,
        temperature_start: float,
        temperature_end: float,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or edges_per_unit < 1:
            raise ValueError("Symbolic-KAN layer dimensions must be positive")
        unknown = sorted(set(primitives) - set(SYMBOLIC_PRIMITIVES))
        if unknown:
            raise ValueError(f"Unknown Symbolic-KAN primitives: {unknown}")
        if not primitives:
            raise ValueError("Symbolic-KAN needs at least one primitive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.edges_per_unit = int(edges_per_unit)
        self.primitives = tuple(primitives)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.temperature = float(temperature_start)
        self.hard = False

        shape = (out_features, edges_per_unit)
        primitive_shape = (*shape, len(primitives))
        self.projection_weight = nn.Parameter(
            torch.empty(out_features, edges_per_unit, in_features)
        )
        self.projection_bias = nn.Parameter(torch.zeros(*shape))
        self.interaction_projection_weight = nn.Parameter(
            torch.empty(out_features, edges_per_unit, in_features)
        )
        self.interaction_projection_bias = nn.Parameter(torch.zeros(*shape))
        self.primitive_logits = nn.Parameter(torch.zeros(*primitive_shape))
        self.gamma = nn.Parameter(torch.ones(*primitive_shape))
        self.beta = nn.Parameter(torch.zeros(*primitive_shape))
        self.interaction_gamma = nn.Parameter(torch.ones(*primitive_shape))
        self.interaction_beta = nn.Parameter(torch.zeros(*primitive_shape))
        self.amplitude = nn.Parameter(torch.empty(*primitive_shape))
        self.output_bias = nn.Parameter(torch.zeros(*primitive_shape))
        self.unit_logits = nn.Parameter(torch.full((out_features,), 2.0))

        self.register_buffer(
            "hard_primitive_index",
            torch.zeros(*shape, dtype=torch.long),
        )
        self.register_buffer(
            "hard_edge_index",
            torch.zeros(out_features, dtype=torch.long),
        )
        self.register_buffer(
            "hard_unit_mask",
            torch.ones(out_features),
        )
        self.register_buffer(
            "hard_projection_mask",
            torch.ones(out_features, edges_per_unit, in_features),
        )
        self.register_buffer(
            "hard_interaction_projection_mask",
            torch.ones(out_features, edges_per_unit, in_features),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection_weight)
        nn.init.xavier_uniform_(self.interaction_projection_weight)
        nn.init.normal_(self.primitive_logits, mean=0.0, std=0.02)
        nn.init.normal_(self.amplitude, mean=0.0, std=0.2)

    def set_progress(self, progress: float) -> None:
        progress = min(max(float(progress), 0.0), 1.0)
        if self.temperature_start <= 0 or self.temperature_end <= 0:
            raise ValueError("Symbolic-KAN temperatures must be positive")
        ratio = self.temperature_end / self.temperature_start
        self.temperature = self.temperature_start * ratio**progress

    def primitive_probabilities(self) -> torch.Tensor:
        return torch.softmax(
            self.primitive_logits / max(self.temperature, 1e-6),
            dim=-1,
        )

    def _primitive_weights(self) -> torch.Tensor:
        if self.hard:
            return F.one_hot(
                self.hard_primitive_index,
                num_classes=len(self.primitives),
            ).to(self.projection_weight.dtype)
        if self.training:
            return F.gumbel_softmax(
                self.primitive_logits,
                tau=max(self.temperature, 1e-6),
                hard=False,
                dim=-1,
            )
        return self.primitive_probabilities()

    def _edge_weights(self, primitive_weights: torch.Tensor) -> torch.Tensor:
        if self.hard:
            return F.one_hot(
                self.hard_edge_index,
                num_classes=self.edges_per_unit,
            ).to(self.projection_weight.dtype)
        confidence = primitive_weights.max(dim=-1).values
        scores = torch.softmax(confidence, dim=-1)
        return _straight_through_top1(scores)

    def _unit_weights(self) -> torch.Tensor:
        if self.hard:
            return self.hard_unit_mask
        return torch.sigmoid(self.unit_logits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projection_weight = self.projection_weight
        interaction_projection_weight = self.interaction_projection_weight
        if self.hard:
            projection_weight = projection_weight * self.hard_projection_mask
            interaction_projection_weight = (
                interaction_projection_weight
                * self.hard_interaction_projection_mask
            )
        scalar_projection = torch.einsum(
            "bi,kei->bke",
            inputs,
            projection_weight,
        ) + self.projection_bias
        interaction_scalar_projection = torch.einsum(
            "bi,kei->bke",
            inputs,
            interaction_projection_weight,
        ) + self.interaction_projection_bias
        primitive_input = (
            scalar_projection.unsqueeze(-1) * self.gamma
            + self.beta
        )
        interaction_primitive_input = (
            interaction_scalar_projection.unsqueeze(-1) * self.interaction_gamma
            + self.interaction_beta
        )
        primitives = _primitive_library(
            primitive_input,
            self.primitives,
            interaction_primitive_input,
        )
        primitive_weights = self._primitive_weights()
        edge_values = torch.sum(
            primitive_weights
            * (self.amplitude * primitives + self.output_bias),
            dim=-1,
        )
        edge_weights = self._edge_weights(primitive_weights)
        unit_values = torch.sum(edge_weights * edge_values, dim=-1)
        return unit_values * self._unit_weights()

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        probabilities = self.primitive_probabilities().clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
        if self.edges_per_unit > 1:
            gram = torch.einsum("kep,kfp->kef", probabilities, probabilities)
            upper = torch.triu(
                torch.ones_like(gram, dtype=torch.bool),
                diagonal=1,
            )
            nms = gram[upper].mean()
        else:
            nms = probabilities.new_zeros(())
        unit_density = torch.sigmoid(self.unit_logits).mean()
        return {
            "entropy": entropy,
            "nms": nms,
            "unit_density": unit_density,
            "bias": self.output_bias.square().mean(),
            "projection_l1": 0.5
            * (
                self.projection_weight.abs().mean()
                + self.interaction_projection_weight.abs().mean()
            ),
        }

    @torch.no_grad()
    def harden(
        self,
        *,
        unit_threshold: float,
        projection_top_k: int,
    ) -> None:
        probabilities = self.primitive_probabilities()
        self.hard_primitive_index.copy_(probabilities.argmax(dim=-1))
        confidence = probabilities.max(dim=-1).values
        self.hard_edge_index.copy_(confidence.argmax(dim=-1))

        units = (torch.sigmoid(self.unit_logits) > unit_threshold).to(
            self.hard_unit_mask.dtype
        )
        if torch.count_nonzero(units) == 0:
            units[torch.argmax(self.unit_logits)] = 1.0
        self.hard_unit_mask.copy_(units)

        top_k = min(max(int(projection_top_k), 1), self.in_features)
        mask = torch.zeros_like(self.hard_projection_mask)
        interaction_mask = torch.zeros_like(
            self.hard_interaction_projection_mask
        )
        for unit_index in range(self.out_features):
            edge_index = int(self.hard_edge_index[unit_index])
            weights = self.projection_weight[unit_index, edge_index].abs()
            selected = torch.topk(weights, k=top_k).indices
            mask[unit_index, edge_index, selected] = 1.0
            primitive_index = int(
                self.hard_primitive_index[unit_index, edge_index].item()
            )
            if self.primitives[primitive_index] == "product":
                interaction_weights = self.interaction_projection_weight[
                    unit_index, edge_index
                ].abs()
                interaction_selected = torch.topk(
                    interaction_weights,
                    k=top_k,
                ).indices
                interaction_mask[
                    unit_index,
                    edge_index,
                    interaction_selected,
                ] = 1.0
        self.hard_projection_mask.copy_(mask)
        self.hard_interaction_projection_mask.copy_(interaction_mask)
        self.hard = True

    def gate_parameters(self) -> list[nn.Parameter]:
        return [self.primitive_logits, self.unit_logits]

    def selected_unit(self, unit_index: int) -> dict[str, Any]:
        edge_index = int(self.hard_edge_index[unit_index].item())
        primitive_index = int(
            self.hard_primitive_index[unit_index, edge_index].item()
        )
        live_projection = self.hard_projection_mask[
            unit_index, edge_index
        ].bool()
        record = {
            "alive": bool(self.hard_unit_mask[unit_index].item()),
            "edge_index": edge_index,
            "primitive_index": primitive_index,
            "primitive": self.primitives[primitive_index],
            "projection_weight": self.projection_weight[
                unit_index, edge_index
            ][live_projection].detach().cpu().tolist(),
            "projection_indices": torch.where(live_projection)[0].cpu().tolist(),
            "projection_bias": float(
                self.projection_bias[unit_index, edge_index].detach().cpu()
            ),
            "gamma": float(
                self.gamma[unit_index, edge_index, primitive_index].detach().cpu()
            ),
            "beta": float(
                self.beta[unit_index, edge_index, primitive_index].detach().cpu()
            ),
            "amplitude": float(
                self.amplitude[
                    unit_index, edge_index, primitive_index
                ].detach().cpu()
            ),
            "output_bias": float(
                self.output_bias[
                    unit_index, edge_index, primitive_index
                ].detach().cpu()
            ),
        }
        if self.primitives[primitive_index] == "product":
            live_interaction_projection = self.hard_interaction_projection_mask[
                unit_index, edge_index
            ].bool()
            record.update(
                {
                    "interaction_projection_weight": (
                        self.interaction_projection_weight[
                            unit_index, edge_index
                        ][live_interaction_projection].detach().cpu().tolist()
                    ),
                    "interaction_projection_indices": torch.where(
                        live_interaction_projection
                    )[0].cpu().tolist(),
                    "interaction_projection_bias": float(
                        self.interaction_projection_bias[
                            unit_index, edge_index
                        ].detach().cpu()
                    ),
                    "interaction_gamma": float(
                        self.interaction_gamma[
                            unit_index, edge_index, primitive_index
                        ].detach().cpu()
                    ),
                    "interaction_beta": float(
                        self.interaction_beta[
                            unit_index, edge_index, primitive_index
                        ].detach().cpu()
                    ),
                }
            )
        return record


class SymbolicKANSingleOutput(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        edges_per_unit: int,
        primitives: Sequence[str],
        *,
        temperature_start: float,
        temperature_end: float,
        regularization: SymbolicRegularization,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("Symbolic-KAN requires at least one symbolic layer")
        dims = [in_features, *[int(value) for value in hidden_dims]]
        self.layers = nn.ModuleList(
            [
                SymbolicKANLayer(
                    src,
                    dst,
                    edges_per_unit,
                    primitives,
                    temperature_start=temperature_start,
                    temperature_end=temperature_end,
                )
                for src, dst in zip(dims[:-1], dims[1:])
            ]
        )
        self.regularization = regularization
        self.progress = 0.0
        self.global_feature_top_k: int | None = None
        self.register_buffer(
            "hard_global_feature_mask",
            torch.ones(in_features),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for layer in self.layers:
            values = layer(values)
        return values.sum(dim=-1, keepdim=True)

    def set_progress(self, progress: float) -> None:
        self.progress = min(max(float(progress), 0.0), 1.0)
        for layer in self.layers:
            layer.set_progress(self.progress)

    def symbolic_regularization(self) -> torch.Tensor:
        entropy = next(self.parameters()).new_zeros(())
        nms = entropy.clone()
        unit = entropy.clone()
        bias = entropy.clone()
        projection_l1 = entropy.clone()
        for layer in self.layers:
            terms = layer.regularization_terms()
            entropy = entropy + terms["entropy"]
            nms = nms + terms["nms"]
            unit = unit + (
                terms["unit_density"] - self.regularization.target_density
            ).square()
            bias = bias + terms["bias"]
            projection_l1 = projection_l1 + terms["projection_l1"]
        selection_schedule = self.regularization.selection * self.progress
        return (
            selection_schedule
            * (
                self.regularization.entropy * entropy
                + self.regularization.nms * nms
            )
            + self.regularization.unit * unit
            + self.regularization.bias * bias
            + self.regularization.projection_l1 * projection_l1
        )

    @torch.no_grad()
    def harden(
        self,
        *,
        unit_threshold: float,
        projection_top_k: int,
        global_feature_top_k: int | None = None,
    ) -> None:
        for layer_index, layer in enumerate(self.layers):
            layer.harden(
                unit_threshold=unit_threshold,
                projection_top_k=(
                    layer.in_features
                    if layer_index == 0 and global_feature_top_k is not None
                    else projection_top_k
                ),
            )
        self.global_feature_top_k = (
            None
            if global_feature_top_k is None
            else min(max(int(global_feature_top_k), 1), self.layers[0].in_features)
        )
        if self.global_feature_top_k is None:
            self.hard_global_feature_mask.fill_(1.0)
            return

        first_layer = self.layers[0]
        importance = first_layer.projection_weight.new_zeros(first_layer.in_features)
        for unit_index in range(first_layer.out_features):
            if not bool(first_layer.hard_unit_mask[unit_index].item()):
                continue
            edge_index = int(first_layer.hard_edge_index[unit_index].item())
            importance.add_(
                first_layer.projection_weight[unit_index, edge_index].abs()
            )
            primitive_index = int(
                first_layer.hard_primitive_index[
                    unit_index, edge_index
                ].item()
            )
            if first_layer.primitives[primitive_index] == "product":
                importance.add_(
                    first_layer.interaction_projection_weight[
                        unit_index, edge_index
                    ].abs()
                )
        selected = torch.topk(
            importance,
            k=self.global_feature_top_k,
        ).indices
        self.hard_global_feature_mask.zero_()
        self.hard_global_feature_mask[selected] = 1.0

        shared_mask = torch.zeros_like(first_layer.hard_projection_mask)
        shared_interaction_mask = torch.zeros_like(
            first_layer.hard_interaction_projection_mask
        )
        local_top_k = min(max(int(projection_top_k), 1), len(selected))
        for unit_index in range(first_layer.out_features):
            if not bool(first_layer.hard_unit_mask[unit_index].item()):
                continue
            edge_index = int(first_layer.hard_edge_index[unit_index].item())
            local_weights = first_layer.projection_weight[
                unit_index, edge_index, selected
            ].abs()
            local_selected = selected[
                torch.topk(local_weights, k=local_top_k).indices
            ]
            shared_mask[unit_index, edge_index, local_selected] = 1.0
            primitive_index = int(
                first_layer.hard_primitive_index[
                    unit_index, edge_index
                ].item()
            )
            if first_layer.primitives[primitive_index] == "product":
                interaction_weights = first_layer.interaction_projection_weight[
                    unit_index, edge_index, selected
                ].abs()
                interaction_selected = selected[
                    torch.topk(interaction_weights, k=local_top_k).indices
                ]
                shared_interaction_mask[
                    unit_index,
                    edge_index,
                    interaction_selected,
                ] = 1.0
        first_layer.hard_projection_mask.copy_(shared_mask)
        first_layer.hard_interaction_projection_mask.copy_(
            shared_interaction_mask
        )

    def selected_input_indices(self) -> list[int]:
        if self.global_feature_top_k is not None:
            return torch.where(self.hard_global_feature_mask.bool())[0].cpu().tolist()
        first_layer = self.layers[0]
        selected: set[int] = set()
        for unit_index in range(first_layer.out_features):
            if not bool(first_layer.hard_unit_mask[unit_index].item()):
                continue
            edge_index = int(first_layer.hard_edge_index[unit_index].item())
            live = first_layer.hard_projection_mask[unit_index, edge_index].bool()
            selected.update(torch.where(live)[0].cpu().tolist())
            primitive_index = int(
                first_layer.hard_primitive_index[
                    unit_index, edge_index
                ].item()
            )
            if first_layer.primitives[primitive_index] == "product":
                interaction_live = first_layer.hard_interaction_projection_mask[
                    unit_index, edge_index
                ].bool()
                selected.update(
                    torch.where(interaction_live)[0].cpu().tolist()
                )
        return sorted(selected)

    def gate_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for layer in self.layers
            for parameter in layer.gate_parameters()
        ]


class SymbolicKAN(nn.Module):
    """Independent paper-style Symbolic-KAN formula for every regression target."""

    def __init__(
        self,
        in_features: int,
        target_names: Sequence[str],
        hidden_dims: Sequence[int] = (4,),
        edges_per_unit: int = 3,
        primitives: Sequence[str] = SYMBOLIC_PRIMITIVES,
        *,
        temperature_start: float = 2.0,
        temperature_end: float = 0.1,
        regularization: SymbolicRegularization | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.target_names = [str(name) for name in target_names]
        self.hidden_dims = [int(value) for value in hidden_dims]
        self.edges_per_unit = int(edges_per_unit)
        self.primitives = tuple(primitives)
        self.networks = nn.ModuleList(
            [
                SymbolicKANSingleOutput(
                    in_features,
                    self.hidden_dims,
                    edges_per_unit,
                    self.primitives,
                    temperature_start=temperature_start,
                    temperature_end=temperature_end,
                    regularization=regularization or SymbolicRegularization(),
                )
                for _ in self.target_names
            ]
        )
        self.architecture = "paper-symbolic-kan"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = [network(inputs) for network in self.networks]
        values = torch.cat(outputs, dim=-1)
        return values[:, 0] if values.shape[1] == 1 else values

    def set_progress(self, progress: float) -> None:
        for network in self.networks:
            network.set_progress(progress)

    def symbolic_regularization(self) -> torch.Tensor:
        return torch.stack(
            [network.symbolic_regularization() for network in self.networks]
        ).mean()

    @torch.no_grad()
    def harden(
        self,
        *,
        unit_threshold: float,
        projection_top_k: int,
        global_feature_top_k: int | None = None,
    ) -> None:
        for network in self.networks:
            network.harden(
                unit_threshold=unit_threshold,
                projection_top_k=projection_top_k,
                global_feature_top_k=global_feature_top_k,
            )

    def gate_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for network in self.networks
            for parameter in network.gate_parameters()
        ]

    def continuous_parameters(self) -> list[nn.Parameter]:
        gate_ids = {id(parameter) for parameter in self.gate_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in gate_ids
        ]


def _primitive_expression(
    name: str,
    argument: str,
    interaction_argument: str | None = None,
) -> str:
    if name == "product":
        if interaction_argument is None:
            raise ValueError("Product expression requires a second argument")
        return f"({argument})*({interaction_argument})"
    return {
        "zero": "0",
        "one": "1",
        "identity": argument,
        "square": f"({argument})^2",
        "cube": f"({argument})^3",
        "sin": f"sin({argument})",
        "cos": f"cos({argument})",
        "tanh": f"tanh({argument})",
        "exp": f"protected_exp({argument})",
        "log1p_abs": f"log(1+abs({argument}))",
        "lorentz": f"1/(1+({argument})^2)",
        "gaussian": f"exp(-({argument})^2)",
        "sinh": f"protected_sinh({argument})",
        "cosh": f"protected_cosh({argument})",
    }[name]


def export_symbolic_kan(
    model: SymbolicKAN,
    feature_names: Sequence[str],
    target_means: Sequence[float] | None = None,
    target_stds: Sequence[float] | None = None,
    feature_scales: Sequence[float] | None = None,
    feature_offsets: Sequence[float] | None = None,
    feature_impute_values: Sequence[float] | None = None,
) -> tuple[dict[str, Any], str]:
    if len(feature_names) != model.in_features:
        raise ValueError("Feature-name count does not match Symbolic-KAN input")
    export_raw_inputs = feature_scales is not None or feature_offsets is not None
    if (feature_scales is None) != (feature_offsets is None):
        raise ValueError("Feature scales and offsets must be provided together")
    scales = (
        [float(value) for value in feature_scales]
        if feature_scales is not None
        else [1.0] * model.in_features
    )
    offsets = (
        [float(value) for value in feature_offsets]
        if feature_offsets is not None
        else [0.0] * model.in_features
    )
    impute_values = (
        [float(value) for value in feature_impute_values]
        if feature_impute_values is not None
        else [float("nan")] * model.in_features
    )
    if (
        len(scales) != model.in_features
        or len(offsets) != model.in_features
        or len(impute_values) != model.in_features
    ):
        raise ValueError(
            "Feature preprocessing parameter count does not match Symbolic-KAN input"
        )
    payload: dict[str, Any] = {
        "method": "symbolic_kan_discrete_gated",
        "paper": "arXiv:2603.23854",
        "adaptation": (
            "projection top-k is applied at hardening for readable "
            "high-dimensional MODNet descriptor formulas"
        ),
        "architecture": [
            model.in_features,
            *model.hidden_dims,
            "sum",
        ],
        "edges_per_unit": model.edges_per_unit,
        "global_feature_top_k": (
            model.networks[0].global_feature_top_k
            if len(model.networks) == 1
            else [network.global_feature_top_k for network in model.networks]
        ),
        "primitive_library": list(model.primitives),
        "input_space": (
            "raw_descriptors" if export_raw_inputs else "preprocessed_descriptors"
        ),
        "input_preprocessing_folded": export_raw_inputs,
        "targets": [],
    }
    report = [
        "Symbolic-KAN discrete formulas",
        "method = arXiv:2603.23854 gated primitives + hardening",
        "",
    ]
    means = list(target_means) if target_means is not None else [0.0] * len(model.target_names)
    stds = list(target_stds) if target_stds is not None else [1.0] * len(model.target_names)
    if len(means) != len(model.target_names) or len(stds) != len(model.target_names):
        raise ValueError("Target scaling count does not match Symbolic-KAN targets")
    for target_index, (target_name, network) in enumerate(
        zip(model.target_names, model.networks)
    ):
        previous_names = [f"x{index}" for index in range(model.in_features)]
        definitions = []
        operators: set[str] = set()
        active_features: set[str] = set()
        layer_records = []
        for layer_index, layer in enumerate(network.layers):
            current_names = [f"h{layer_index}_{index}" for index in range(layer.out_features)]
            unit_records = []
            for unit_index, variable in enumerate(current_names):
                selected = layer.selected_unit(unit_index)
                if not selected["alive"]:
                    unit_records.append({"variable": variable, **selected, "expression": "0"})
                    definitions.append({"variable": variable, "expression": "0"})
                    continue
                exported = dict(selected)
                projection_indices = [
                    int(index) for index in selected["projection_indices"]
                ]
                model_projection_weights = [
                    float(value) for value in selected["projection_weight"]
                ]
                projection_weights = list(model_projection_weights)
                projection_bias = float(selected["projection_bias"])
                if layer_index == 0 and export_raw_inputs:
                    projection_weights = [
                        weight * scales[source_index]
                        for source_index, weight in zip(
                            projection_indices,
                            model_projection_weights,
                        )
                    ]
                    projection_bias += sum(
                        weight * offsets[source_index]
                        for source_index, weight in zip(
                            projection_indices,
                            model_projection_weights,
                        )
                    )
                    exported.update(
                        {
                            "projection_weight": projection_weights,
                            "projection_bias": projection_bias,
                            "projection_input_space": "raw_descriptors",
                            "preprocessed_projection_weight": model_projection_weights,
                            "preprocessed_projection_bias": float(
                                selected["projection_bias"]
                            ),
                        }
                    )
                projection_terms = []
                for source_index, coefficient in zip(
                    projection_indices,
                    projection_weights,
                ):
                    source = previous_names[int(source_index)]
                    projection_terms.append(f"{float(coefficient):.8g}*{source}")
                    if layer_index == 0:
                        active_features.add(str(feature_names[int(source_index)]))
                projection = " + ".join(projection_terms)
                projection += f"{projection_bias:+.8g}"
                inner = (
                    f"{float(selected['gamma']):.8g}*({projection})"
                    f"{float(selected['beta']):+.8g}"
                )
                interaction_inner = None
                if str(selected["primitive"]) == "product":
                    interaction_indices = [
                        int(index)
                        for index in selected["interaction_projection_indices"]
                    ]
                    model_interaction_weights = [
                        float(value)
                        for value in selected[
                            "interaction_projection_weight"
                        ]
                    ]
                    interaction_weights = list(model_interaction_weights)
                    interaction_bias = float(
                        selected["interaction_projection_bias"]
                    )
                    if layer_index == 0 and export_raw_inputs:
                        interaction_weights = [
                            weight * scales[source_index]
                            for source_index, weight in zip(
                                interaction_indices,
                                model_interaction_weights,
                            )
                        ]
                        interaction_bias += sum(
                            weight * offsets[source_index]
                            for source_index, weight in zip(
                                interaction_indices,
                                model_interaction_weights,
                            )
                        )
                        exported.update(
                            {
                                "interaction_projection_weight": (
                                    interaction_weights
                                ),
                                "interaction_projection_bias": (
                                    interaction_bias
                                ),
                                "interaction_projection_input_space": (
                                    "raw_descriptors"
                                ),
                                "preprocessed_interaction_projection_weight": (
                                    model_interaction_weights
                                ),
                                "preprocessed_interaction_projection_bias": float(
                                    selected[
                                        "interaction_projection_bias"
                                    ]
                                ),
                            }
                        )
                    interaction_terms = []
                    for source_index, coefficient in zip(
                        interaction_indices,
                        interaction_weights,
                    ):
                        source = previous_names[int(source_index)]
                        interaction_terms.append(
                            f"{float(coefficient):.8g}*{source}"
                        )
                        if layer_index == 0:
                            active_features.add(
                                str(feature_names[int(source_index)])
                            )
                    interaction_projection = " + ".join(interaction_terms)
                    interaction_projection += f"{interaction_bias:+.8g}"
                    interaction_inner = (
                        f"{float(selected['interaction_gamma']):.8g}"
                        f"*({interaction_projection})"
                        f"{float(selected['interaction_beta']):+.8g}"
                    )
                primitive = _primitive_expression(
                    str(selected["primitive"]),
                    inner,
                    interaction_inner,
                )
                expression = (
                    f"{float(selected['amplitude']):.8g}*({primitive})"
                    f"{float(selected['output_bias']):+.8g}"
                )
                operators.add(str(selected["primitive"]))
                record = {
                    "variable": variable,
                    **exported,
                    "expression": expression,
                }
                unit_records.append(record)
                definitions.append(
                    {"variable": variable, "expression": expression}
                )
            layer_records.append(
                {
                    "layer": layer_index,
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "units": unit_records,
                }
            )
            previous_names = current_names

        final_layer = network.layers[-1]
        live_outputs = [
            previous_names[index]
            for index in range(final_layer.out_features)
            if bool(final_layer.hard_unit_mask[index].item())
        ]
        scaled_expression = " + ".join(live_outputs) if live_outputs else "0"
        expression = (
            f"{float(stds[target_index]):.8g}*({scaled_expression})"
            f"{float(means[target_index]):+.8g}"
        )
        variable_definitions = []
        for index, name in enumerate(feature_names):
            if str(name) not in active_features:
                continue
            definition: dict[str, Any] = {
                "variable": f"x{index}",
                "feature": str(name),
                "expression": (
                    f"raw_descriptor({name!r})"
                    if export_raw_inputs
                    else f"preprocessed({name!r})"
                ),
                "input_space": (
                    "raw_descriptor"
                    if export_raw_inputs
                    else "preprocessed_descriptor"
                ),
            }
            if export_raw_inputs:
                definition.update(
                    {
                        "impute_value": impute_values[index],
                        "folded_scale": scales[index],
                        "folded_offset": offsets[index],
                    }
                )
            variable_definitions.append(definition)
        target_record = {
            "target": target_name,
            "expression": expression,
            "scaled_network_expression": scaled_expression,
            "target_mean": float(means[target_index]),
            "target_std": float(stds[target_index]),
            "hidden_definitions": definitions,
            "variable_definitions": variable_definitions,
            "active_feature_names": sorted(active_features),
            "selected_global_feature_indices": network.selected_input_indices(),
            "selected_global_feature_names": [
                str(feature_names[index])
                for index in network.selected_input_indices()
            ],
            "operators": sorted(operators),
            "term_names": sorted(operators),
            "layers": layer_records,
            "n_active_units": sum(
                int(unit["alive"])
                for layer in layer_records
                for unit in layer["units"]
            ),
        }
        payload["targets"].append(target_record)
        report.append(f"target = {target_name}")
        for definition in variable_definitions:
            variable_line = (
                f"  {definition['variable']} = {definition['expression']}"
            )
            impute_value = definition.get("impute_value")
            if impute_value is not None and impute_value == impute_value:
                variable_line += (
                    f"  [missing -> training impute value "
                    f"{float(impute_value):.8g}]"
                )
            report.append(variable_line)
        for definition in definitions:
            report.append(
                f"  {definition['variable']} = {definition['expression']}"
            )
        report.append(f"  y = {expression}")
        report.append("")
    return payload, "\n".join(report)
