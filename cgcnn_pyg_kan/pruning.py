from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

from .kan import FastKANLinear, KANLinear


def iter_kan_modules(model: torch.nn.Module) -> Iterable[KANLinear | FastKANLinear]:
    for module in model.modules():
        if isinstance(module, (KANLinear, FastKANLinear)):
            yield module


def iter_kan_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    seen: set[int] = set()
    for module in iter_kan_modules(model):
        if isinstance(module, KANLinear):
            parameters = (module.base_weight, module.spline_weight, module.spline_scaler)
        else:
            parameters = (module.spline_linear.weight,) + (
                (module.base_linear.weight,) if module.base_linear is not None else ()
            )
        for parameter in parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter


def kan_l1_penalty(model: torch.nn.Module) -> torch.Tensor:
    parameters = list(iter_kan_parameters(model))
    if not parameters:
        return next(model.parameters()).new_zeros(())
    return torch.stack([parameter.abs().mean() for parameter in parameters]).mean()


@dataclass
class PruningMasks:
    """Persistent zero masks used during post-pruning fine-tuning."""

    masks: list[tuple[torch.nn.Parameter, torch.Tensor]] = field(default_factory=list)
    handles: list[Any] = field(default_factory=list)
    pruned_edges: int = 0
    total_edges: int = 0

    @property
    def pruned_parameters(self) -> int:
        return int(sum(torch.count_nonzero(~mask).item() for _, mask in self.masks))

    def install(self) -> None:
        self.enforce()
        for parameter, mask in self.masks:
            self.handles.append(parameter.register_hook(lambda grad, mask=mask: grad * mask))

    @torch.no_grad()
    def enforce(self) -> None:
        for parameter, mask in self.masks:
            parameter.mul_(mask)

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _edge_scores(
    model: torch.nn.Module,
) -> list[tuple[float, KANLinear | FastKANLinear, int, int]]:
    edges: list[tuple[float, KANLinear | FastKANLinear, int, int]] = []
    for module in iter_kan_modules(model):
        if isinstance(module, KANLinear):
            spline = module.scaled_spline_weight.detach()
            base = module.base_weight.detach()
            scores = torch.sqrt(base.square() + spline.square().sum(dim=-1))
        else:
            grids = int(module.rbf.grid.numel())
            spline = module.spline_linear.weight.detach().view(
                module.out_features, module.in_features, grids
            )
            scores = spline.square().sum(dim=-1)
            if module.base_linear is not None:
                scores = scores + module.base_linear.weight.detach().square()
            scores = torch.sqrt(scores)
        for output_idx in range(module.out_features):
            for input_idx in range(module.in_features):
                edges.append(
                    (float(scores[output_idx, input_idx].cpu()), module, output_idx, input_idx)
                )
    return edges


def _mask_parameter(
    masks: dict[torch.nn.Parameter, torch.Tensor],
    parameter: torch.nn.Parameter,
) -> torch.Tensor:
    if parameter not in masks:
        masks[parameter] = torch.ones_like(parameter, dtype=torch.bool)
    return masks[parameter]


def apply_kan_edge_pruning(model: torch.nn.Module, fraction: float) -> PruningMasks:
    """Globally prune low-norm KAN edges without disconnecting output nodes."""

    if not 0.0 <= fraction < 1.0:
        raise ValueError("--prune-kan-fraction must be in [0, 1)")
    all_edges = _edge_scores(model)
    result = PruningMasks(total_edges=len(all_edges))
    if fraction <= 0 or not all_edges:
        return result

    protected: set[tuple[int, int, int]] = set()
    grouped: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for score, module, output_idx, input_idx in all_edges:
        grouped.setdefault((id(module), output_idx), []).append((score, input_idx))
    for (module_id, output_idx), values in grouped.items():
        protected.add((module_id, output_idx, max(values)[1]))

    candidates = [
        edge
        for edge in all_edges
        if (id(edge[1]), edge[2], edge[3]) not in protected
    ]
    n_prune = min(int(round(len(all_edges) * fraction)), len(candidates))
    selected = sorted(candidates, key=lambda edge: edge[0])[:n_prune]
    masks: dict[torch.nn.Parameter, torch.Tensor] = {}
    for _, module, output_idx, input_idx in selected:
        if isinstance(module, KANLinear):
            _mask_parameter(masks, module.base_weight)[output_idx, input_idx] = False
            _mask_parameter(masks, module.spline_weight)[output_idx, input_idx, :] = False
            _mask_parameter(masks, module.spline_scaler)[output_idx, input_idx] = False
        else:
            grids = int(module.rbf.grid.numel())
            start = input_idx * grids
            stop = start + grids
            _mask_parameter(masks, module.spline_linear.weight)[output_idx, start:stop] = False
            if module.base_linear is not None:
                _mask_parameter(masks, module.base_linear.weight)[output_idx, input_idx] = False

    result.masks = list(masks.items())
    result.pruned_edges = len(selected)
    result.install()
    return result


def apply_kan_parameter_pruning(model: torch.nn.Module, fraction: float) -> PruningMasks:
    """Scalar KAN-only magnitude pruning retained as an ablation."""

    if not 0.0 <= fraction < 1.0:
        raise ValueError("--prune-kan-fraction must be in [0, 1)")
    parameters = [parameter for parameter in iter_kan_parameters(model) if parameter.ndim >= 2]
    result = PruningMasks()
    total = sum(parameter.numel() for parameter in parameters)
    n_prune = min(int(round(total * fraction)), max(0, total - 1))
    if n_prune <= 0:
        return result
    scores = torch.cat([parameter.detach().abs().flatten().cpu() for parameter in parameters])
    selected = torch.topk(scores, k=n_prune, largest=False).indices
    masks = []
    offset = 0
    for parameter in parameters:
        mask = torch.ones_like(parameter, dtype=torch.bool)
        stop = offset + parameter.numel()
        local = selected[(selected >= offset) & (selected < stop)] - offset
        mask.view(-1)[local.to(mask.device)] = False
        masks.append((parameter, mask))
        offset = stop
    result.masks = masks
    result.install()
    return result


def apply_kan_pruning(model: torch.nn.Module, fraction: float, mode: str) -> PruningMasks:
    if mode == "edge":
        return apply_kan_edge_pruning(model, fraction)
    if mode == "parameter":
        return apply_kan_parameter_pruning(model, fraction)
    raise ValueError(f"unsupported pruning mode {mode!r}")
