from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class KANLinear(nn.Module):
    """A compact B-spline KAN layer.

    The layer combines a standard base branch with per-input univariate spline
    bases. Keeping the base branch makes the module usable even when activations
    move outside the spline grid early in training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        base_activation: type[nn.Module] = nn.SiLU,
        spline_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if spline_order < 1:
            raise ValueError("spline_order must be at least 1")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("grid_range must be an increasing pair")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        step = (grid_range[1] - grid_range[0]) / grid_size
        knots = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
            * step
            + grid_range[0]
        )
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())

        self.base_activation = base_activation()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_scale = spline_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=self.spline_scale)
        nn.init.ones_(self.spline_scaler)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"expected input shape [batch, {self.in_features}], got {tuple(x.shape)}"
            )

        grid = self.grid.to(device=x.device, dtype=x.dtype)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for order in range(1, self.spline_order + 1):
            left_num = x - grid[:, : -(order + 1)]
            left_den = grid[:, order:-1] - grid[:, : -(order + 1)]
            right_num = grid[:, order + 1 :] - x
            right_den = grid[:, order + 1 :] - grid[:, 1:-order]

            bases = (
                left_num / left_den.clamp_min(torch.finfo(x.dtype).eps) * bases[:, :, :-1]
                + right_num / right_den.clamp_min(torch.finfo(x.dtype).eps) * bases[:, :, 1:]
            )

        return bases.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_basis = self.b_splines(x).reshape(x.size(0), -1)
        spline_weight = self.scaled_spline_weight.reshape(self.out_features, -1)
        return base_output + F.linear(spline_basis, spline_weight)


class KANMLP(nn.Module):
    """Small KAN network used as a drop-in replacement for CGCNN MLP blocks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] | int,
        out_dim: int,
        dropout: float = 0.0,
        grid_size: int = 3,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        dims = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []
        for idx, (src, dst) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(
                KANLinear(
                    src,
                    dst,
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
            )
            if idx < len(dims) - 2:
                layers.append(nn.LayerNorm(dst))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
