from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

from .kan import KANMLP


class CrystalGraphConv(MessagePassing):
    """CGCNN-style gated crystal graph convolution for PyG Data objects."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        conv_net: Literal["mlp", "kan"] = "mlp",
        conv_mlp_hidden_dim: int | None = None,
        conv_kan_hidden_dim: int = 16,
        conv_kan_grid_size: int = 3,
        conv_kan_spline_order: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr="add")
        pair_dim = 2 * node_dim + edge_dim
        output_dim = 2 * node_dim
        if conv_net == "mlp":
            mlp_hidden_dim = conv_mlp_hidden_dim or output_dim
            self.interaction_net = nn.Sequential(
                nn.Linear(pair_dim, mlp_hidden_dim),
                nn.Softplus(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(mlp_hidden_dim, output_dim),
            )
        elif conv_net == "kan":
            self.interaction_net = KANMLP(
                pair_dim,
                conv_kan_hidden_dim,
                output_dim,
                dropout=dropout,
                grid_size=conv_kan_grid_size,
                spline_order=conv_kan_spline_order,
            )
        else:
            raise ValueError(f"unsupported conv_net {conv_net!r}; expected 'mlp' or 'kan'")

        self.bn_message = nn.BatchNorm1d(node_dim)
        self.bn_update = nn.BatchNorm1d(node_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        message = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        message = self.bn_message(message)
        return F.softplus(self.bn_update(x + message))

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        gate, core = self.interaction_net(z).chunk(2, dim=-1)
        return torch.sigmoid(gate) * F.softplus(core)


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []
        for idx, (src, dst) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(src, dst))
            if idx < len(dims) - 2:
                layers.append(nn.Softplus())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CGCNN(nn.Module):
    """PyG CGCNN with switchable MLP or KAN interaction networks."""

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 64,
        num_convs: int = 3,
        head_hidden_dims: Sequence[int] = (128, 64),
        out_dim: int = 1,
        conv_net: Literal["mlp", "kan"] = "mlp",
        conv_mlp_hidden_dim: int | None = None,
        conv_kan_hidden_dim: int = 16,
        conv_kan_grid_size: int = 3,
        conv_kan_spline_order: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_convs < 1:
            raise ValueError("num_convs must be at least 1")

        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            CrystalGraphConv(
                hidden_dim,
                edge_input_dim,
                conv_net=conv_net,
                conv_mlp_hidden_dim=conv_mlp_hidden_dim,
                conv_kan_hidden_dim=conv_kan_hidden_dim,
                conv_kan_grid_size=conv_kan_grid_size,
                conv_kan_spline_order=conv_kan_spline_order,
                dropout=dropout,
            )
            for _ in range(num_convs)
        )
        self.head = MLPHead(hidden_dim, head_hidden_dims, out_dim, dropout)

    def forward(self, data) -> torch.Tensor:
        x = F.softplus(self.node_embedding(data.x))
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
        pooled = global_mean_pool(x, data.batch)
        return self.head(pooled).squeeze(-1)
