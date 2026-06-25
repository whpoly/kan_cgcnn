from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class SyntheticConfig:
    num_graphs: int = 512
    node_dim: int = 10
    edge_dim: int = 16
    min_nodes: int = 8
    max_nodes: int = 24
    cutoff: float = 0.65
    noise_std: float = 0.02
    seed: int = 7


def _edge_features(distances: torch.Tensor, edge_dim: int) -> torch.Tensor:
    centers = torch.linspace(0.0, 1.5, edge_dim, dtype=distances.dtype)
    width = 1.5 / edge_dim
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (width**2))


def make_synthetic_crystal_dataset(config: SyntheticConfig) -> list[Data]:
    """Create deterministic crystal-like graphs for quick model benchmarks."""

    generator = torch.Generator().manual_seed(config.seed)
    dataset: list[Data] = []
    for _ in range(config.num_graphs):
        num_nodes = int(
            torch.randint(
                config.min_nodes,
                config.max_nodes + 1,
                size=(1,),
                generator=generator,
            )
        )
        x = torch.randn(num_nodes, config.node_dim, generator=generator)
        pos = torch.rand(num_nodes, 3, generator=generator)
        dist = torch.cdist(pos, pos)
        edge_mask = (dist < config.cutoff) & (~torch.eye(num_nodes, dtype=torch.bool))

        if edge_mask.sum() == 0:
            nearest = dist + torch.eye(num_nodes) * 10.0
            src = torch.arange(num_nodes)
            dst = nearest.argmin(dim=1)
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = edge_mask.nonzero(as_tuple=False).t().contiguous()

        edge_dist = dist[edge_index[0], edge_index[1]]
        edge_attr = _edge_features(edge_dist, config.edge_dim)

        atom_term = (
            torch.sin(x[:, 0]).mean()
            + 0.35 * (x[:, 1] * x[:, 2]).mean()
            + 0.15 * torch.cos(x[:, 3]).mean()
        )
        pair_term = 0.25 * torch.exp(-edge_dist).mean()
        size_term = 0.05 * math.log(num_nodes)
        noise = config.noise_std * torch.randn((), generator=generator)
        y = (atom_term + pair_term + size_term + noise).view(1)

        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos))

    return dataset
