from __future__ import annotations

from collections import OrderedDict
from typing import Literal, Sequence

import torch
from torch import nn

from .kan import make_kan_mlp

TargetHierarchy = Sequence[Sequence[Sequence[str]]]
BlockType = Literal["kan", "mlp"]
HeadType = Literal["linear", "kan"]
ActivationName = Literal["relu", "elu", "silu"]


def flatten_targets(targets: TargetHierarchy) -> list[str]:
    return [name for group in targets for prop in group for name in prop]


def _default_targets() -> list[list[list[str]]]:
    return [[["target"]]]


def _as_dim_list(values: Sequence[int] | int | None) -> list[int]:
    if values is None:
        return []
    if isinstance(values, int):
        return [values]
    return list(values)


def _normalize_num_neurons(
    num_neurons: Sequence[Sequence[int] | int | None] | None,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if num_neurons is None:
        num_neurons = ([64], [32], [16], [])
    if len(num_neurons) != 4:
        raise ValueError("num_neurons must contain four blocks: common, group, property, target")
    return tuple(_as_dim_list(block) for block in num_neurons)  # type: ignore[return-value]


def _normalize_block_types(
    block_type: BlockType,
    block_types: Sequence[BlockType] | None,
) -> tuple[BlockType, BlockType, BlockType, BlockType]:
    if block_types is None:
        return (block_type, block_type, block_type, block_type)
    if len(block_types) != 4:
        raise ValueError("block_types must contain common, group, property, and target types")
    normalized = tuple(block_types)
    if any(value not in ("mlp", "kan") for value in normalized):
        raise ValueError("each block type must be 'mlp' or 'kan'")
    return normalized  # type: ignore[return-value]


class IdentityBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for src, dst in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(src, dst))
            layers.append(activation())
            if batch_norm:
                layers.append(nn.BatchNorm1d(dst))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_block(
    in_dim: int,
    hidden_dims: Sequence[int],
    block_type: Literal["kan", "mlp"],
    kan_impl: Literal["spline", "fastkan"],
    kan_grid_size: int,
    kan_spline_order: int,
    kan_use_layernorm: bool,
    dropout: float,
    batch_norm: bool,
    mlp_activation: type[nn.Module],
) -> nn.Module:
    hidden_dims = list(hidden_dims)
    if not hidden_dims:
        return IdentityBlock(in_dim)
    if block_type == "kan":
        block = make_kan_mlp(
            in_dim,
            hidden_dims[:-1],
            hidden_dims[-1],
            impl=kan_impl,
            dropout=dropout,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            use_layernorm=kan_use_layernorm,
        )
        if batch_norm:
            block = nn.Sequential(block, nn.BatchNorm1d(hidden_dims[-1]))
        block.out_dim = hidden_dims[-1]  # type: ignore[attr-defined]
        return block
    if block_type == "mlp":
        return MLPBlock(
            in_dim,
            hidden_dims,
            activation=mlp_activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )
    raise ValueError(f"unsupported block_type {block_type!r}; expected 'kan' or 'mlp'")


def _activation_type(name: ActivationName) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"unsupported MLP activation {name!r}") from exc


def _make_output_head(
    in_dim: int,
    out_dim: int,
    head_type: HeadType,
    kan_impl: Literal["spline", "fastkan"],
    kan_grid_size: int,
    kan_spline_order: int,
    kan_use_layernorm: bool,
) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    if head_type == "kan":
        return make_kan_mlp(
            in_dim,
            [],
            out_dim,
            impl=kan_impl,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            use_layernorm=kan_use_layernorm,
        )
    raise ValueError(f"unsupported output_head_type {head_type!r}")


class MODNetKAN(nn.Module):
    """PyTorch MODNet hierarchy with independently selectable block types.

    MODNet uses a shared dense trunk, group-specific branches, and
    property-specific outputs over material descriptors. ``block_types`` makes
    it possible to retain the MODNet MLP trunk while replacing only the
    target-specific predictor with a KAN. A KAN output head is used in the
    hybrid setup so an empty target block still contains a real KAN mapping.
    """

    def __init__(
        self,
        n_feat: int,
        targets: TargetHierarchy | None = None,
        num_neurons: Sequence[Sequence[int] | int | None] | None = None,
        block_type: BlockType = "kan",
        block_types: Sequence[BlockType] | None = None,
        output_head_type: HeadType = "linear",
        mlp_activation: ActivationName = "relu",
        kan_impl: Literal["spline", "fastkan"] = "fastkan",
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_use_layernorm: bool = True,
        dropout: float = 0.0,
        batch_norm_multi_target: bool = True,
        squeeze_single_target: bool = True,
    ) -> None:
        super().__init__()
        if n_feat < 1:
            raise ValueError("n_feat must be at least 1")
        self.n_feat = n_feat
        self.targets = targets if targets is not None else _default_targets()
        self.target_names = flatten_targets(self.targets)
        if not self.target_names:
            raise ValueError("targets must contain at least one target name")

        self.num_neurons = _normalize_num_neurons(num_neurons)
        self.block_type = block_type
        self.block_types = _normalize_block_types(block_type, block_types)
        self.output_head_type = output_head_type
        self.mlp_activation = mlp_activation
        self.kan_impl = kan_impl
        self.squeeze_single_target = squeeze_single_target
        self._multi_target = len(self.target_names) > 1
        use_batch_norm = batch_norm_multi_target and self._multi_target

        common_dims, group_dims, property_dims, target_dims = self.num_neurons
        common_type, group_type, property_type, target_type = self.block_types
        mlp_activation_type = _activation_type(mlp_activation)
        self.common_block = _make_block(
            n_feat,
            common_dims,
            common_type,
            kan_impl,
            kan_grid_size,
            kan_spline_order,
            kan_use_layernorm,
            dropout,
            use_batch_norm,
            mlp_activation_type,
        )
        common_out_dim = int(self.common_block.out_dim)  # type: ignore[attr-defined]

        self.group_blocks = nn.ModuleDict()
        self.property_blocks = nn.ModuleDict()
        self.target_blocks = nn.ModuleDict()
        self.output_heads = nn.ModuleDict()
        self.output_slices: OrderedDict[str, tuple[int, int]] = OrderedDict()

        cursor = 0
        for group_idx, group in enumerate(self.targets):
            group_key = f"g{group_idx}"
            group_block = _make_block(
                common_out_dim,
                group_dims,
                group_type,
                kan_impl,
                kan_grid_size,
                kan_spline_order,
                kan_use_layernorm,
                dropout,
                use_batch_norm,
                mlp_activation_type,
            )
            self.group_blocks[group_key] = group_block
            group_out_dim = int(group_block.out_dim)  # type: ignore[attr-defined]

            for prop_idx, prop_targets in enumerate(group):
                if not prop_targets:
                    raise ValueError("each property group must contain at least one target name")
                prop_key = f"{group_key}_p{prop_idx}"
                prop_block = _make_block(
                    group_out_dim,
                    property_dims,
                    property_type,
                    kan_impl,
                    kan_grid_size,
                    kan_spline_order,
                    kan_use_layernorm,
                    dropout,
                    use_batch_norm,
                    mlp_activation_type,
                )
                self.property_blocks[prop_key] = prop_block
                prop_out_dim = int(prop_block.out_dim)  # type: ignore[attr-defined]

                target_block = _make_block(
                    prop_out_dim,
                    target_dims,
                    target_type,
                    kan_impl,
                    kan_grid_size,
                    kan_spline_order,
                    kan_use_layernorm,
                    dropout,
                    use_batch_norm,
                    mlp_activation_type,
                )
                self.target_blocks[prop_key] = target_block
                target_out_dim = int(target_block.out_dim)  # type: ignore[attr-defined]
                self.output_heads[prop_key] = _make_output_head(
                    target_out_dim,
                    len(prop_targets),
                    output_head_type,
                    kan_impl,
                    kan_grid_size,
                    kan_spline_order,
                    kan_use_layernorm,
                )
                self.output_slices[prop_key] = (cursor, cursor + len(prop_targets))
                cursor += len(prop_targets)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        common = self.common_block(x)
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for group_idx, group in enumerate(self.targets):
            group_key = f"g{group_idx}"
            group_x = self.group_blocks[group_key](common)
            for prop_idx, _ in enumerate(group):
                prop_key = f"{group_key}_p{prop_idx}"
                prop_x = self.property_blocks[prop_key](group_x)
                target_x = self.target_blocks[prop_key](prop_x)
                outputs[prop_key] = self.output_heads[prop_key](target_x)

        if return_dict:
            return outputs

        y = torch.cat(list(outputs.values()), dim=-1)
        if self.squeeze_single_target and y.size(-1) == 1:
            return y.squeeze(-1)
        return y
