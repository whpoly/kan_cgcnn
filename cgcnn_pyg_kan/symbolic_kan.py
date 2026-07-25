from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
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

DIRECT_SYMBOLIC_PRIMITIVES = (
    "identity",
    "square",
    "tanh",
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
                raise ValueError("Product primitive requires a second feature")
            result = argument * interaction_values[..., index]
        else:
            raise ValueError(f"Unsupported Symbolic-KAN primitive {name!r}")
        outputs.append(
            torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
        )
    return torch.stack(outputs, dim=-1)


@dataclass(frozen=True)
class SymbolicRegularization:
    selection: float = 1e-3
    entropy: float = 1.0
    nms: float = 0.1
    unit: float = 1e-3
    bias: float = 1e-4
    projection_l1: float = 1e-5
    target_density: float = 0.8


class DirectSymbolicLayer(nn.Module):
    """Independent symbolic terms connected directly from features to output."""

    def __init__(
        self,
        in_features: int,
        n_terms: int,
        primitives: Sequence[str],
        *,
        temperature_start: float,
        temperature_end: float,
        hard_start: float,
        max_product_terms: int,
    ) -> None:
        super().__init__()
        if in_features < 1 or n_terms < 1:
            raise ValueError("Direct Symbolic-KAN dimensions must be positive")
        unknown = sorted(set(primitives) - set(SYMBOLIC_PRIMITIVES))
        if unknown:
            raise ValueError(f"Unknown Symbolic-KAN primitives: {unknown}")
        if not primitives:
            raise ValueError("Direct Symbolic-KAN needs at least one primitive")
        if not 0.0 <= hard_start <= 1.0:
            raise ValueError("hard_start must be in [0, 1]")
        if max_product_terms < 0:
            raise ValueError("max_product_terms cannot be negative")

        self.in_features = int(in_features)
        self.out_features = int(n_terms)
        self.n_terms = int(n_terms)
        self.edges_per_unit = 1
        self.primitives = tuple(primitives)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.temperature = float(temperature_start)
        self.hard_start = float(hard_start)
        self.max_product_terms = int(max_product_terms)
        self.progress = 0.0
        self.hard = False

        n_primitives = len(self.primitives)
        self.feature_logits = nn.Parameter(torch.empty(n_terms, in_features))
        self.interaction_feature_logits = nn.Parameter(
            torch.empty(n_terms, in_features)
        )
        self.primitive_logits = nn.Parameter(torch.empty(n_terms, n_primitives))
        self.input_scale = nn.Parameter(torch.ones(n_terms, n_primitives))
        self.input_bias = nn.Parameter(torch.zeros(n_terms, n_primitives))
        self.interaction_scale = nn.Parameter(torch.ones(n_terms, n_primitives))
        self.interaction_bias = nn.Parameter(torch.zeros(n_terms, n_primitives))
        self.amplitude = nn.Parameter(torch.empty(n_terms, n_primitives))
        self.term_bias = nn.Parameter(torch.zeros(n_terms, n_primitives))
        self.unit_logits = nn.Parameter(torch.full((n_terms,), 2.0))
        self.output_bias = nn.Parameter(torch.zeros(()))

        self.register_buffer(
            "hard_feature_index",
            torch.zeros(n_terms, dtype=torch.long),
        )
        self.register_buffer(
            "hard_interaction_feature_index",
            torch.zeros(n_terms, dtype=torch.long),
        )
        self.register_buffer(
            "hard_primitive_index",
            torch.zeros(n_terms, dtype=torch.long),
        )
        self.register_buffer("hard_unit_mask", torch.ones(n_terms))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_logits, mean=0.0, std=0.02)
        nn.init.normal_(self.interaction_feature_logits, mean=0.0, std=0.02)
        nn.init.normal_(self.primitive_logits, mean=0.0, std=0.02)
        nn.init.normal_(self.amplitude, mean=0.0, std=0.2)

    def set_progress(self, progress: float) -> None:
        self.progress = min(max(float(progress), 0.0), 1.0)
        if self.temperature_start <= 0 or self.temperature_end <= 0:
            raise ValueError("Symbolic-KAN temperatures must be positive")
        ratio = self.temperature_end / self.temperature_start
        self.temperature = self.temperature_start * ratio**self.progress

    def _probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)

    def feature_probabilities(self) -> torch.Tensor:
        return self._probabilities(self.feature_logits)

    def interaction_feature_probabilities(self) -> torch.Tensor:
        return self._probabilities(self.interaction_feature_logits)

    def primitive_probabilities(self) -> torch.Tensor:
        return self._probabilities(self.primitive_logits)

    def _soft_or_straight_through(
        self,
        logits: torch.Tensor,
        hard_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.hard:
            return F.one_hot(
                hard_indices,
                num_classes=logits.shape[-1],
            ).to(logits.dtype)
        use_hard_sample = self.training and self.progress >= self.hard_start
        if self.training:
            return F.gumbel_softmax(
                logits,
                tau=max(self.temperature, 1e-6),
                hard=use_hard_sample,
                dim=-1,
            )
        return self._probabilities(logits)

    def _unit_weights(self) -> torch.Tensor:
        if self.hard:
            return self.hard_unit_mask
        return torch.sigmoid(self.unit_logits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_weights = self._soft_or_straight_through(
            self.feature_logits,
            self.hard_feature_index,
        )
        interaction_feature_weights = self._soft_or_straight_through(
            self.interaction_feature_logits,
            self.hard_interaction_feature_index,
        )
        primitive_weights = self._soft_or_straight_through(
            self.primitive_logits,
            self.hard_primitive_index,
        )
        selected = inputs @ feature_weights.transpose(0, 1)
        interaction_selected = (
            inputs @ interaction_feature_weights.transpose(0, 1)
        )
        primitive_inputs = (
            selected.unsqueeze(-1) * self.input_scale.unsqueeze(0)
            + self.input_bias.unsqueeze(0)
        )
        interaction_inputs = (
            interaction_selected.unsqueeze(-1)
            * self.interaction_scale.unsqueeze(0)
            + self.interaction_bias.unsqueeze(0)
        )
        primitive_values = _primitive_library(
            primitive_inputs,
            self.primitives,
            interaction_inputs,
        )
        term_values = torch.sum(
            primitive_weights.unsqueeze(0)
            * (
                self.amplitude.unsqueeze(0) * primitive_values
                + self.term_bias.unsqueeze(0)
            ),
            dim=-1,
        )
        return self.output_bias + torch.sum(
            term_values * self._unit_weights().unsqueeze(0),
            dim=-1,
            keepdim=True,
        )

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        primitive = self.primitive_probabilities().clamp_min(1e-12)
        feature = self.feature_probabilities().clamp_min(1e-12)
        interaction = self.interaction_feature_probabilities().clamp_min(1e-12)
        entropy = (
            -(primitive * primitive.log()).sum(dim=-1).mean()
            -(feature * feature.log()).sum(dim=-1).mean()
        )
        if "product" in self.primitives:
            product_index = self.primitives.index("product")
            product_probability = primitive[:, product_index].mean()
            entropy = entropy + product_probability * (
                -(interaction * interaction.log()).sum(dim=-1).mean()
            )
        if self.n_terms > 1:
            feature_overlap = feature @ feature.transpose(0, 1)
            upper = torch.triu(
                torch.ones_like(feature_overlap, dtype=torch.bool),
                diagonal=1,
            )
            nms = feature_overlap[upper].mean()
        else:
            nms = feature.new_zeros(())
        if "product" in self.primitives:
            product_index = self.primitives.index("product")
            product_excess = torch.relu(
                primitive[:, product_index].sum() - self.max_product_terms
            )
            nms = nms + product_excess.square()
        return {
            "entropy": entropy,
            "nms": nms,
            "unit_density": torch.sigmoid(self.unit_logits).mean(),
            "bias": self.term_bias.square().mean() + self.output_bias.square(),
            "projection_l1": 0.5
            * (self.amplitude.abs().mean() + self.input_scale.abs().mean()),
        }

    @torch.no_grad()
    def harden(
        self,
        *,
        unit_threshold: float,
        global_feature_top_k: int | None = None,
    ) -> None:
        primitive_probabilities = self.primitive_probabilities()
        primitive_indices = primitive_probabilities.argmax(dim=-1)

        if "product" in self.primitives:
            product_index = self.primitives.index("product")
            product_terms = torch.where(primitive_indices == product_index)[0]
            if len(product_terms) > self.max_product_terms:
                confidence = primitive_probabilities[
                    product_terms, product_index
                ]
                keep = product_terms[
                    torch.topk(confidence, k=self.max_product_terms).indices
                ]
                keep_mask = torch.zeros(
                    self.n_terms,
                    dtype=torch.bool,
                    device=primitive_indices.device,
                )
                keep_mask[keep] = True
                non_product_indices = [
                    index
                    for index, name in enumerate(self.primitives)
                    if name != "product"
                ]
                if not non_product_indices:
                    primitive_indices[product_terms[~keep_mask[product_terms]]] = (
                        product_index
                    )
                else:
                    alternatives = primitive_probabilities[
                        :, non_product_indices
                    ].argmax(dim=-1)
                    for term_index in product_terms:
                        if not bool(keep_mask[term_index]):
                            primitive_indices[term_index] = non_product_indices[
                                int(alternatives[term_index])
                            ]
        self.hard_primitive_index.copy_(primitive_indices)

        units = (torch.sigmoid(self.unit_logits) > unit_threshold).to(
            self.hard_unit_mask.dtype
        )
        if torch.count_nonzero(units) == 0:
            units[torch.argmax(self.unit_logits)] = 1.0
        self.hard_unit_mask.copy_(units)

        feature_probabilities = self.feature_probabilities()
        interaction_probabilities = self.interaction_feature_probabilities()
        candidate_indices = torch.arange(
            self.in_features,
            device=feature_probabilities.device,
        )
        if global_feature_top_k is not None:
            top_k = min(max(int(global_feature_top_k), 1), self.in_features)
            importance = torch.sum(
                feature_probabilities
                * torch.sigmoid(self.unit_logits).unsqueeze(-1),
                dim=0,
            )
            if "product" in self.primitives:
                product_index = self.primitives.index("product")
                product_weight = primitive_probabilities[:, product_index]
                importance = importance + torch.sum(
                    interaction_probabilities
                    * product_weight.unsqueeze(-1)
                    * torch.sigmoid(self.unit_logits).unsqueeze(-1),
                    dim=0,
                )
            candidate_indices = torch.topk(importance, k=top_k).indices

        for term_index in range(self.n_terms):
            primary_local = torch.argmax(
                feature_probabilities[term_index, candidate_indices]
            )
            primary = candidate_indices[primary_local]
            self.hard_feature_index[term_index] = primary

            interaction_scores = interaction_probabilities[
                term_index, candidate_indices
            ].clone()
            if len(candidate_indices) > 1:
                interaction_scores[candidate_indices == primary] = -1.0
            secondary_local = torch.argmax(interaction_scores)
            self.hard_interaction_feature_index[term_index] = (
                candidate_indices[secondary_local]
            )
        self.hard = True

    def gate_parameters(self) -> list[nn.Parameter]:
        return [
            self.feature_logits,
            self.interaction_feature_logits,
            self.primitive_logits,
            self.unit_logits,
        ]

    def selected_input_indices(self) -> list[int]:
        selected: set[int] = set()
        for term_index in range(self.n_terms):
            if not bool(self.hard_unit_mask[term_index].item()):
                continue
            selected.add(int(self.hard_feature_index[term_index].item()))
            primitive = self.primitives[
                int(self.hard_primitive_index[term_index].item())
            ]
            if primitive == "product":
                selected.add(
                    int(
                        self.hard_interaction_feature_index[
                            term_index
                        ].item()
                    )
                )
        return sorted(selected)

    def selected_term(self, term_index: int) -> dict[str, Any]:
        primitive_index = int(self.hard_primitive_index[term_index].item())
        primitive = self.primitives[primitive_index]
        record: dict[str, Any] = {
            "alive": bool(self.hard_unit_mask[term_index].item()),
            "term_index": term_index,
            "primitive_index": primitive_index,
            "primitive": primitive,
            "feature_index": int(self.hard_feature_index[term_index].item()),
            "input_scale": float(
                self.input_scale[term_index, primitive_index].detach().cpu()
            ),
            "input_bias": float(
                self.input_bias[term_index, primitive_index].detach().cpu()
            ),
            "amplitude": float(
                self.amplitude[term_index, primitive_index].detach().cpu()
            ),
            "term_bias": float(
                self.term_bias[term_index, primitive_index].detach().cpu()
            ),
        }
        if primitive == "product":
            record.update(
                {
                    "interaction_feature_index": int(
                        self.hard_interaction_feature_index[
                            term_index
                        ].item()
                    ),
                    "interaction_scale": float(
                        self.interaction_scale[
                            term_index, primitive_index
                        ].detach().cpu()
                    ),
                    "interaction_bias": float(
                        self.interaction_bias[
                            term_index, primitive_index
                        ].detach().cpu()
                    ),
                }
            )
        return record


class DirectSymbolicNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_terms: int,
        primitives: Sequence[str],
        *,
        temperature_start: float,
        temperature_end: float,
        hard_start: float,
        max_product_terms: int,
        regularization: SymbolicRegularization,
    ) -> None:
        super().__init__()
        self.layer = DirectSymbolicLayer(
            in_features,
            n_terms,
            primitives,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            hard_start=hard_start,
            max_product_terms=max_product_terms,
        )
        self.layers = nn.ModuleList([self.layer])
        self.regularization = regularization
        self.progress = 0.0
        self.global_feature_top_k: int | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)

    def set_progress(self, progress: float) -> None:
        self.progress = min(max(float(progress), 0.0), 1.0)
        self.layer.set_progress(self.progress)

    def symbolic_regularization(self) -> torch.Tensor:
        terms = self.layer.regularization_terms()
        selection_schedule = self.regularization.selection * self.progress
        unit = (
            terms["unit_density"] - self.regularization.target_density
        ).square()
        return (
            selection_schedule
            * (
                self.regularization.entropy * terms["entropy"]
                + self.regularization.nms * terms["nms"]
            )
            + self.regularization.unit * unit
            + self.regularization.bias * terms["bias"]
            + self.regularization.projection_l1 * terms["projection_l1"]
        )

    @torch.no_grad()
    def harden(
        self,
        *,
        unit_threshold: float,
        projection_top_k: int = 1,
        global_feature_top_k: int | None = None,
    ) -> None:
        if projection_top_k != 1:
            raise ValueError(
                "Direct Symbolic-KAN selects exactly one feature per unary term"
            )
        self.global_feature_top_k = (
            None
            if global_feature_top_k is None
            else min(max(int(global_feature_top_k), 1), self.layer.in_features)
        )
        self.layer.harden(
            unit_threshold=unit_threshold,
            global_feature_top_k=self.global_feature_top_k,
        )

    def selected_input_indices(self) -> list[int]:
        return self.layer.selected_input_indices()


class SymbolicKAN(nn.Module):
    """Direct additive Symbolic-KAN with no compositional hidden layers."""

    def __init__(
        self,
        in_features: int,
        target_names: Sequence[str],
        hidden_dims: Sequence[int] = (5,),
        edges_per_unit: int = 1,
        primitives: Sequence[str] = DIRECT_SYMBOLIC_PRIMITIVES,
        *,
        temperature_start: float = 2.0,
        temperature_end: float = 0.2,
        hard_start: float = 0.7,
        max_product_terms: int = 1,
        regularization: SymbolicRegularization | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = [int(value) for value in hidden_dims]
        if len(hidden_dims) != 1 or hidden_dims[0] < 1:
            raise ValueError(
                "Direct Symbolic-KAN expects one positive term count, "
                "for example hidden_dims=[5]"
            )
        if edges_per_unit != 1:
            raise ValueError(
                "Direct Symbolic-KAN has one direct feature selection per term; "
                "edges_per_unit must be 1"
            )
        if not target_names:
            raise ValueError("Symbolic-KAN needs at least one target")
        self.in_features = int(in_features)
        self.architecture = "direct-additive-symbolic-kan"
        self.target_names = [str(name) for name in target_names]
        self.hidden_dims = hidden_dims
        self.edges_per_unit = 1
        self.primitives = tuple(primitives)
        self.hard_start = float(hard_start)
        self.max_product_terms = int(max_product_terms)
        symbolic_regularization = regularization or SymbolicRegularization()
        self.networks = nn.ModuleList(
            [
                DirectSymbolicNetwork(
                    self.in_features,
                    hidden_dims[0],
                    self.primitives,
                    temperature_start=temperature_start,
                    temperature_end=temperature_end,
                    hard_start=hard_start,
                    max_product_terms=max_product_terms,
                    regularization=symbolic_regularization,
                )
                for _ in self.target_names
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = [network(inputs) for network in self.networks]
        result = torch.cat(outputs, dim=-1)
        return result.squeeze(-1) if len(self.target_names) == 1 else result

    def set_progress(self, progress: float) -> None:
        for network in self.networks:
            network.set_progress(progress)

    def symbolic_regularization(self) -> torch.Tensor:
        return torch.stack(
            [network.symbolic_regularization() for network in self.networks]
        ).sum()

    def harden(
        self,
        *,
        unit_threshold: float,
        projection_top_k: int = 1,
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
            for parameter in network.layer.gate_parameters()
        ]

    def continuous_parameters(self) -> list[nn.Parameter]:
        gate_ids = {id(parameter) for parameter in self.gate_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in gate_ids
        ]


def _primitive_expression(
    primitive: str,
    inner: str,
    interaction_inner: str | None = None,
) -> str:
    expressions = {
        "zero": "0",
        "one": "1",
        "identity": f"({inner})",
        "square": f"({inner})^2",
        "cube": f"({inner})^3",
        "sin": f"sin({inner})",
        "cos": f"cos({inner})",
        "tanh": f"tanh({inner})",
        "exp": f"exp(clip({inner},-10,10))",
        "log1p_abs": f"log(1+abs({inner}))",
        "lorentz": f"1/(1+({inner})^2)",
        "gaussian": f"exp(-clip(({inner})^2,0,20))",
        "sinh": f"sinh(clip({inner},-5,5))",
        "cosh": f"cosh(clip({inner},-5,5))",
    }
    if primitive == "product":
        if interaction_inner is None:
            raise ValueError("Product expression requires a second feature")
        return f"({inner})*({interaction_inner})"
    return expressions[primitive]


def export_symbolic_kan(
    model: SymbolicKAN,
    feature_names: Sequence[str],
    *,
    target_means: Sequence[float] | None = None,
    target_stds: Sequence[float] | None = None,
    feature_scales: Sequence[float] | None = None,
    feature_offsets: Sequence[float] | None = None,
    feature_impute_values: Sequence[float] | None = None,
) -> tuple[dict[str, Any], str]:
    if len(feature_names) != model.in_features:
        raise ValueError("Feature names do not match Symbolic-KAN input width")
    if not all(network.layer.hard for network in model.networks):
        raise ValueError("Symbolic-KAN must be hardened before formula export")

    n_targets = len(model.target_names)
    means = np.broadcast_to(
        np.asarray(0.0 if target_means is None else target_means, dtype=float),
        (n_targets,),
    )
    stds = np.broadcast_to(
        np.asarray(1.0 if target_stds is None else target_stds, dtype=float),
        (n_targets,),
    )
    export_raw_inputs = feature_scales is not None or feature_offsets is not None
    if export_raw_inputs and (
        feature_scales is None or feature_offsets is None
    ):
        raise ValueError(
            "feature_scales and feature_offsets must be provided together"
        )
    scales = np.ones(model.in_features, dtype=float)
    offsets = np.zeros(model.in_features, dtype=float)
    if export_raw_inputs:
        scales = np.asarray(feature_scales, dtype=float)
        offsets = np.asarray(feature_offsets, dtype=float)
        if len(scales) != model.in_features or len(offsets) != model.in_features:
            raise ValueError("Feature preprocessing width does not match model")
    impute_values = (
        [None] * model.in_features
        if feature_impute_values is None
        else list(feature_impute_values)
    )
    if len(impute_values) != model.in_features:
        raise ValueError("Feature imputation width does not match model")

    payload: dict[str, Any] = {
        "method": "direct_additive_symbolic_kan",
        "architecture": [model.in_features, model.hidden_dims[0], 1],
        "formula_structure": (
            "output_bias + sum_k amplitude_k * "
            "primitive_k(a_k * x_feature_k + b_k)"
        ),
        "input_space": (
            "raw_descriptors" if export_raw_inputs else "preprocessed_descriptors"
        ),
        "primitives": list(model.primitives),
        "edges_per_unit": 1,
        "max_product_terms": model.max_product_terms,
        "hard_start": model.hard_start,
        "global_feature_top_k": (
            model.networks[0].global_feature_top_k
            if len(model.networks) == 1
            else [
                network.global_feature_top_k for network in model.networks
            ]
        ),
        "targets": [],
    }
    report = [
        "Direct additive Symbolic-KAN formulas",
        "structure = raw features -> independent symbolic terms -> output sum",
        "no compositional hidden layers",
        "",
    ]

    for target_index, (target_name, network) in enumerate(
        zip(model.target_names, model.networks)
    ):
        layer = network.layer
        active_features: set[int] = set()
        operators: set[str] = set()
        term_records: list[dict[str, Any]] = []
        term_expressions: list[str] = []
        combined_bias = float(layer.output_bias.detach().cpu())
        for term_index in range(layer.n_terms):
            selected = layer.selected_term(term_index)
            if not selected["alive"]:
                continue
            feature_index = int(selected["feature_index"])
            primitive = str(selected["primitive"])
            active_features.add(feature_index)
            operators.add(primitive)

            raw_scale = float(selected["input_scale"]) * scales[feature_index]
            raw_bias = (
                float(selected["input_scale"]) * offsets[feature_index]
                + float(selected["input_bias"])
            )
            inner = f"{raw_scale:.8g}*x{feature_index}{raw_bias:+.8g}"
            interaction_inner = None
            exported = {
                **selected,
                "feature": str(feature_names[feature_index]),
                "raw_input_scale": raw_scale,
                "raw_input_bias": raw_bias,
            }
            if primitive == "product":
                interaction_index = int(
                    selected["interaction_feature_index"]
                )
                active_features.add(interaction_index)
                interaction_scale = (
                    float(selected["interaction_scale"])
                    * scales[interaction_index]
                )
                interaction_bias = (
                    float(selected["interaction_scale"])
                    * offsets[interaction_index]
                    + float(selected["interaction_bias"])
                )
                interaction_inner = (
                    f"{interaction_scale:.8g}*x{interaction_index}"
                    f"{interaction_bias:+.8g}"
                )
                exported.update(
                    {
                        "interaction_feature": str(
                            feature_names[interaction_index]
                        ),
                        "raw_interaction_scale": interaction_scale,
                        "raw_interaction_bias": interaction_bias,
                    }
                )
            body = _primitive_expression(
                primitive,
                inner,
                interaction_inner,
            )
            expression = (
                f"{float(selected['amplitude']):.8g}*({body})"
                f"{float(selected['term_bias']):+.8g}"
            )
            combined_bias += float(selected["term_bias"])
            exported["expression"] = expression
            exported["variable"] = f"term{term_index}"
            term_records.append(exported)
            term_expressions.append(
                f"{float(selected['amplitude']):.8g}*({body})"
            )

        scaled_sum = f"{combined_bias:.8g}"
        if term_expressions:
            scaled_sum += " + " + " + ".join(term_expressions)
        target_expression = (
            f"{float(stds[target_index]):.8g}*({scaled_sum})"
            f"{float(means[target_index]):+.8g}"
        )
        variable_definitions = []
        for feature_index in sorted(active_features):
            definition: dict[str, Any] = {
                "variable": f"x{feature_index}",
                "feature": str(feature_names[feature_index]),
                "expression": (
                    f"raw_descriptor({feature_names[feature_index]!r})"
                    if export_raw_inputs
                    else f"preprocessed({feature_names[feature_index]!r})"
                ),
                "input_space": (
                    "raw_descriptor"
                    if export_raw_inputs
                    else "preprocessed_descriptor"
                ),
            }
            impute_value = impute_values[feature_index]
            if impute_value is not None:
                definition["impute_value"] = float(impute_value)
            variable_definitions.append(definition)

        target_record = {
            "target": target_name,
            "expression": target_expression,
            "scaled_network_expression": scaled_sum,
            "target_mean": float(means[target_index]),
            "target_std": float(stds[target_index]),
            "hidden_definitions": [],
            "formula_terms": term_records,
            "variable_definitions": variable_definitions,
            "active_feature_names": [
                str(feature_names[index]) for index in sorted(active_features)
            ],
            "selected_global_feature_indices": sorted(active_features),
            "selected_global_feature_names": [
                str(feature_names[index]) for index in sorted(active_features)
            ],
            "operators": sorted(operators),
            "term_names": sorted(operators),
            "layers": [
                {
                    "layer": 0,
                    "kind": "direct_additive_terms",
                    "in_features": layer.in_features,
                    "out_features": layer.n_terms,
                    "units": term_records,
                }
            ],
            "n_active_units": len(term_records),
            "n_formula_terms": len(term_records),
        }
        payload["targets"].append(target_record)

        report.append(f"target = {target_name}")
        for definition in variable_definitions:
            line = f"  {definition['variable']} = {definition['expression']}"
            if "impute_value" in definition:
                line += (
                    "  [missing -> training impute value "
                    f"{float(definition['impute_value']):.8g}]"
                )
            report.append(line)
        report.append(f"  y = {target_expression}")
        report.append("")

    return payload, "\n".join(report)
