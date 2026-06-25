from __future__ import annotations

from typing import Literal, Sequence

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


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float | None = None,
    ) -> None:
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid.to(device=x.device, dtype=x.dtype)
        return torch.exp(-(((x.unsqueeze(-1) - grid) / self.denominator) ** 2))


class FastKANLinear(nn.Module):
    """FastKAN layer using Gaussian RBF bases plus a base linear branch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_layernorm: bool = True,
        use_base_update: bool = True,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layernorm = nn.LayerNorm(in_features) if use_layernorm and in_features > 1 else None
        self.rbf = RadialBasisFunction(grid_min=grid_min, grid_max=grid_max, num_grids=num_grids)
        self.spline_linear = nn.Linear(in_features * num_grids, out_features, bias=False)
        nn.init.trunc_normal_(
            self.spline_linear.weight,
            mean=0.0,
            std=spline_weight_init_scale,
        )
        self.use_base_update = use_base_update
        self.base_linear = nn.Linear(in_features, out_features) if use_base_update else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spline_input = self.layernorm(x) if self.layernorm is not None else x
        spline_basis = self.rbf(spline_input).flatten(start_dim=-2)
        output = self.spline_linear(spline_basis)
        if self.base_linear is not None:
            output = output + self.base_linear(F.silu(x))
        return output


class FastKANMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] | int,
        out_dim: int,
        dropout: float = 0.0,
        num_grids: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        dims = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []
        for idx, (src, dst) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(
                FastKANLinear(
                    src,
                    dst,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                )
            )
            if idx < len(dims) - 2 and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_kan_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    impl: Literal["spline", "fastkan"] = "fastkan",
    dropout: float = 0.0,
    grid_size: int = 8,
    spline_order: int = 3,
) -> nn.Module:
    if impl == "spline":
        return KANMLP(
            in_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            grid_size=grid_size,
            spline_order=spline_order,
        )
    if impl == "fastkan":
        return FastKANMLP(
            in_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            num_grids=grid_size,
        )
    raise ValueError(f"unsupported KAN implementation {impl!r}")
