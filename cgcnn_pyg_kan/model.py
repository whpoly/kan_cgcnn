from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

from .kan import make_kan_mlp


class CrystalGraphConv(MessagePassing):
    """CGCNN-style gated crystal graph convolution for PyG Data objects."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        conv_net: Literal["mlp", "kan"] = "mlp",
        conv_kan_impl: Literal["spline", "fastkan"] = "fastkan",
        conv_kan_hidden_dim: int = 16,
        conv_kan_grid_size: int = 8,
        conv_kan_spline_order: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr="add")
        pair_dim = 2 * node_dim + edge_dim
        output_dim = 2 * node_dim
        if conv_net == "mlp":
            self.fc_full = nn.Linear(pair_dim, output_dim)
        elif conv_net == "kan":
            self.fc_full = make_kan_mlp(
                pair_dim,
                conv_kan_hidden_dim,
                output_dim,
                impl=conv_kan_impl,
                dropout=dropout,
                grid_size=conv_kan_grid_size,
                spline_order=conv_kan_spline_order,
            )
        else:
            raise ValueError(f"unsupported conv_net {conv_net!r}; expected 'mlp' or 'kan'")

        self.bn1 = nn.BatchNorm1d(output_dim)
        self.bn2 = nn.BatchNorm1d(node_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        message = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        message = self.bn2(message)
        return F.softplus(x + message)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        gate, core = self.bn1(self.fc_full(z)).chunk(2, dim=-1)
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
        num_convs: int = 4,
        head_hidden_dims: Sequence[int] = (32,),
        out_dim: int = 1,
        conv_net: Literal["mlp", "kan"] = "mlp",
        conv_kan_impl: Literal["spline", "fastkan"] = "fastkan",
        conv_kan_hidden_dim: int = 16,
        conv_kan_grid_size: int = 8,
        conv_kan_spline_order: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_convs < 1:
            raise ValueError("num_convs must be at least 1")

        self.embedding = nn.Linear(node_input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            CrystalGraphConv(
                hidden_dim,
                edge_input_dim,
                conv_net=conv_net,
                conv_kan_impl=conv_kan_impl,
                conv_kan_hidden_dim=conv_kan_hidden_dim,
                conv_kan_grid_size=conv_kan_grid_size,
                conv_kan_spline_order=conv_kan_spline_order,
                dropout=dropout,
            )
            for _ in range(num_convs)
        )
        if len(head_hidden_dims) < 1:
            raise ValueError("head_hidden_dims must contain at least h_fea_len")
        self.conv_to_fc = nn.Linear(hidden_dim, head_hidden_dims[0])
        self.conv_to_fc_softplus = nn.Softplus()
        self.fcs = nn.ModuleList(
            nn.Linear(src, dst) for src, dst in zip(head_hidden_dims[:-1], head_hidden_dims[1:])
        )
        self.softpluses = nn.ModuleList(nn.Softplus() for _ in self.fcs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc_out = nn.Linear(head_hidden_dims[-1], out_dim)

    def forward(self, data) -> torch.Tensor:
        x = self.embedding(data.x)
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
        pooled = global_mean_pool(x, data.batch)
        out = self.conv_to_fc_softplus(self.conv_to_fc(pooled))
        out = self.dropout(out)
        for fc, softplus in zip(self.fcs, self.softpluses):
            out = softplus(fc(out))
            out = self.dropout(out)
        return self.fc_out(out).squeeze(-1)
