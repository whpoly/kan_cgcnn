from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Iterable, Literal, Sequence

import torch
from torch_geometric.data import Data

CGCNN_ATOM_FEATURE_DIM = 92
ELEMENTAL_FEATURE_NAMES = (
    "atomic_number",
    "group",
    "row",
    "atomic_mass",
    "electronegativity",
    "atomic_radius",
    "atomic_radius_calculated",
    "mendeleev_number",
)


@dataclass(frozen=True)
class StructureGraphConfig:
    cutoff: float = 6.0
    edge_dim: int = 41
    max_atomic_number: int = 92
    gaussian_width: float | None = None
    atom_features: Literal["onehot", "atomic_number", "elemental", "cgcnn"] = "onehot"
    edge_features: Literal["gaussian", "distance"] = "gaussian"

    def __post_init__(self) -> None:
        if self.atom_features not in ("onehot", "atomic_number", "elemental", "cgcnn"):
            raise ValueError("atom_features must be 'onehot', 'atomic_number', 'elemental', or 'cgcnn'")
        if self.edge_features not in ("gaussian", "distance"):
            raise ValueError("edge_features must be 'gaussian' or 'distance'")
        if self.edge_dim < 1:
            raise ValueError("edge_dim must be at least 1")

    @property
    def node_dim(self) -> int:
        if self.atom_features == "atomic_number":
            return 1
        if self.atom_features == "elemental":
            return len(ELEMENTAL_FEATURE_NAMES)
        if self.atom_features == "cgcnn":
            return CGCNN_ATOM_FEATURE_DIM
        return self.max_atomic_number

    @property
    def edge_input_dim(self) -> int:
        if self.edge_features == "distance":
            return 1
        return self.edge_dim


def gaussian_distance_features(
    distances: torch.Tensor,
    cutoff: float,
    edge_dim: int,
    width: float | None = None,
) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, edge_dim, dtype=distances.dtype)
    width = width or cutoff / edge_dim
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (width**2))


def atom_features(
    elements: Sequence,
    config: StructureGraphConfig,
) -> torch.Tensor:
    atomic_numbers = [int(element.Z) for element in elements]
    if config.atom_features == "atomic_number":
        return torch.tensor(atomic_numbers, dtype=torch.float32).view(-1, 1)
    if config.atom_features == "elemental":
        return torch.stack(
            [_elemental_feature_vector(element, config) for element in elements],
            dim=0,
        )
    if config.atom_features == "cgcnn":
        return torch.stack(
            [_cgcnn_atom_feature_vector(atomic_number) for atomic_number in atomic_numbers],
            dim=0,
        )

    x = torch.zeros(len(atomic_numbers), config.node_dim, dtype=torch.float32)
    for idx, atomic_number in enumerate(atomic_numbers):
        clipped = min(max(atomic_number, 1), config.max_atomic_number)
        x[idx, clipped - 1] = 1.0
    return x


def distance_features(
    distances: torch.Tensor,
    config: StructureGraphConfig,
) -> torch.Tensor:
    if config.edge_features == "distance":
        return distances.view(-1, 1)
    return gaussian_distance_features(
        distances,
        cutoff=config.cutoff,
        edge_dim=config.edge_dim,
        width=config.gaussian_width,
    )


def structure_to_graph(
    structure,
    target: float | int | None = None,
    config: StructureGraphConfig = StructureGraphConfig(),
) -> Data:
    """Convert a pymatgen Structure into a directed periodic PyG graph."""

    elements = [_site_element(site) for site in structure]
    x = atom_features(elements, config)

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
    edge_attr = distance_features(distances, config)
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


def _elemental_feature_vector(element, config: StructureGraphConfig) -> torch.Tensor:
    atomic_number = min(max(float(element.Z), 1.0), float(config.max_atomic_number))
    group = _safe_float(getattr(element, "group", None))
    row = _safe_float(getattr(element, "row", None))
    atomic_mass = _safe_float(getattr(element, "atomic_mass", None))
    electronegativity = _safe_float(getattr(element, "X", None))
    atomic_radius = _safe_float(getattr(element, "atomic_radius", None))
    atomic_radius_calculated = _safe_float(getattr(element, "atomic_radius_calculated", None))
    mendeleev_number = _safe_float(getattr(element, "mendeleev_no", None))

    values = [
        atomic_number / float(config.max_atomic_number),
        group / 18.0,
        row / 7.0,
        atomic_mass / 250.0,
        electronegativity / 4.0,
        atomic_radius / 3.0,
        atomic_radius_calculated / 3.0,
        mendeleev_number / 103.0,
    ]
    return torch.tensor(values, dtype=torch.float32)


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if torch.isfinite(torch.tensor(number)):
        return number
    return 0.0


@lru_cache(maxsize=1)
def _cgcnn_atom_feature_table() -> dict[int, torch.Tensor]:
    try:
        resource = files("matminer.utils.data_files").joinpath("cgcnn_atom_feature.json")
        data = json.loads(resource.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "atom_features='cgcnn' requires matminer and its cgcnn_atom_feature.json data file"
        ) from exc
    return {
        int(atomic_number): torch.tensor(features, dtype=torch.float32)
        for atomic_number, features in data.items()
    }


def _cgcnn_atom_feature_vector(atomic_number: int) -> torch.Tensor:
    table = _cgcnn_atom_feature_table()
    if atomic_number in table:
        return table[atomic_number]
    clipped = min(max(atomic_number, min(table)), max(table))
    return table.get(clipped, torch.zeros(CGCNN_ATOM_FEATURE_DIM, dtype=torch.float32))


def _site_element(site):
    try:
        return site.specie
    except AttributeError:
        species = site.species
        element, _ = max(species.items(), key=lambda item: item[1])
        return element


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
