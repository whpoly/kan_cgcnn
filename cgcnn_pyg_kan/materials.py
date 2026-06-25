from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class StructureGraphConfig:
    cutoff: float = 6.0
    edge_dim: int = 41
    max_atomic_number: int = 92
    gaussian_width: float | None = None

    @property
    def node_dim(self) -> int:
        return self.max_atomic_number


def gaussian_distance_features(
    distances: torch.Tensor,
    cutoff: float,
    edge_dim: int,
    width: float | None = None,
) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, edge_dim, dtype=distances.dtype)
    width = width or cutoff / edge_dim
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (width**2))


def structure_to_graph(
    structure,
    target: float | int | None = None,
    config: StructureGraphConfig = StructureGraphConfig(),
) -> Data:
    """Convert a pymatgen Structure into a directed periodic PyG graph."""

    atomic_numbers = [_site_atomic_number(site) for site in structure]
    x = torch.zeros(len(atomic_numbers), config.node_dim, dtype=torch.float32)
    for idx, atomic_number in enumerate(atomic_numbers):
        clipped = min(max(atomic_number, 1), config.max_atomic_number)
        x[idx, clipped - 1] = 1.0

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_distances: list[float] = []
    for src, neighbors in enumerate(structure.get_all_neighbors(config.cutoff)):
        for neighbor in neighbors:
            distance = float(neighbor.nn_distance)
            if distance <= 1e-8:
                continue
            edge_src.append(src)
            edge_dst.append(int(neighbor.index))
            edge_distances.append(distance)

    if not edge_src:
        edge_src, edge_dst, edge_distances = _fallback_edges(structure)

    distances = torch.tensor(edge_distances, dtype=torch.float32)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = gaussian_distance_features(
        distances,
        cutoff=config.cutoff,
        edge_dim=config.edge_dim,
        width=config.gaussian_width,
    )
    pos = torch.tensor(structure.cart_coords, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    if target is not None:
        data.y = torch.tensor([float(target)], dtype=torch.float32)
    return data


def structures_to_graphs(
    structures: Sequence,
    targets: Iterable[float | int] | None,
    config: StructureGraphConfig = StructureGraphConfig(),
) -> list[Data]:
    if targets is None:
        return [structure_to_graph(structure, config=config) for structure in structures]
    return [
        structure_to_graph(structure, target=target, config=config)
        for structure, target in zip(structures, targets)
    ]


def _site_atomic_number(site) -> int:
    try:
        return int(site.specie.Z)
    except AttributeError:
        species = site.species
        element, _ = max(species.items(), key=lambda item: item[1])
        return int(element.Z)


def _fallback_edges(structure) -> tuple[list[int], list[int], list[float]]:
    distance_matrix = structure.distance_matrix
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_distances: list[float] = []
    for src in range(len(structure)):
        distances = [
            (dst, float(distance_matrix[src, dst]))
            for dst in range(len(structure))
            if dst != src and distance_matrix[src, dst] > 1e-8
        ]
        if distances:
            dst, distance = min(distances, key=lambda item: item[1])
            edge_src.extend([src, dst])
            edge_dst.extend([dst, src])
            edge_distances.extend([distance, distance])
    if edge_src:
        return edge_src, edge_dst, edge_distances

    return [0], [0], [1.0]
