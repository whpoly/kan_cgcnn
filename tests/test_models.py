import torch
from torch_geometric.loader import DataLoader

from cgcnn_pyg_kan.data import SyntheticConfig, make_synthetic_crystal_dataset
from cgcnn_pyg_kan.materials import (
    CGCNN_ATOM_FEATURE_DIM,
    ELEMENTAL_FEATURE_NAMES,
    StructureGraphConfig,
    structure_to_graph,
)
from cgcnn_pyg_kan.modnet import MODNetKAN
from cgcnn_pyg_kan.modnet_features import MODNetFeatureProcessor, make_feature_frame
from cgcnn_pyg_kan.model import CGCNN


def test_cgcnn_conv_nets_forward() -> None:
    config = SyntheticConfig(num_graphs=4, seed=11)
    dataset = make_synthetic_crystal_dataset(config)
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    for conv_net in ("mlp", "kan"):
        model = CGCNN(
            node_input_dim=config.node_dim,
            edge_input_dim=config.edge_dim,
            hidden_dim=16,
            num_convs=2,
            head_hidden_dims=(32,),
            conv_net=conv_net,
            head_net="kan" if conv_net == "kan" else "mlp",
            conv_kan_hidden_dim=8,
            conv_kan_grid_size=3,
        )
        output = model(batch)
        assert output.shape == (2,)
        assert torch.isfinite(output).all()


def test_cgcnn_kan_readout_forward() -> None:
    config = SyntheticConfig(num_graphs=4, seed=12)
    dataset = make_synthetic_crystal_dataset(config)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = CGCNN(
        node_input_dim=config.node_dim,
        edge_input_dim=config.edge_dim,
        hidden_dim=16,
        num_convs=2,
        head_hidden_dims=(8,),
        conv_net="mlp",
        head_net="kan",
        head_kan_grid_size=3,
    )
    output = model(batch)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_structure_to_graph_forward() -> None:
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    config = StructureGraphConfig(cutoff=6.0, edge_dim=8)
    graph = structure_to_graph(structure, target=1.0, config=config)
    batch = next(iter(DataLoader([graph], batch_size=1)))
    model = CGCNN(
        node_input_dim=config.node_dim,
        edge_input_dim=config.edge_dim,
        hidden_dim=16,
        num_convs=1,
        head_hidden_dims=(16,),
        conv_net="kan",
        head_net="kan",
        conv_kan_hidden_dim=4,
        conv_kan_grid_size=3,
    )
    output = model(batch)
    assert output.shape == (1,)
    assert graph.edge_attr.shape[1] == config.edge_dim


def test_structure_to_graph_elemental_distance_features_forward() -> None:
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    config = StructureGraphConfig(
        cutoff=6.0,
        atom_features="elemental",
        edge_features="distance",
    )
    graph = structure_to_graph(structure, target=1.0, config=config)
    batch = next(iter(DataLoader([graph], batch_size=1)))
    model = CGCNN(
        node_input_dim=config.node_dim,
        edge_input_dim=config.edge_input_dim,
        hidden_dim=16,
        num_convs=1,
        head_hidden_dims=(16,),
        conv_net="kan",
        head_net="kan",
        conv_kan_hidden_dim=4,
        conv_kan_grid_size=3,
    )
    output = model(batch)
    assert output.shape == (1,)
    assert graph.x.shape[1] == len(ELEMENTAL_FEATURE_NAMES)
    assert graph.edge_attr.shape[1] == 1
    assert torch.isfinite(graph.x).all()
    assert torch.isfinite(graph.edge_attr).all()


def test_structure_to_graph_cgcnn_atom_features_forward() -> None:
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    config = StructureGraphConfig(
        cutoff=6.0,
        edge_dim=8,
        atom_features="cgcnn",
        edge_features="gaussian",
    )
    graph = structure_to_graph(structure, target=1.0, config=config)
    batch = next(iter(DataLoader([graph], batch_size=1)))
    model = CGCNN(
        node_input_dim=config.node_dim,
        edge_input_dim=config.edge_input_dim,
        hidden_dim=16,
        num_convs=1,
        head_hidden_dims=(16,),
        conv_net="mlp",
    )
    output = model(batch)
    assert output.shape == (1,)
    assert graph.x.shape[1] == CGCNN_ATOM_FEATURE_DIM
    assert graph.edge_attr.shape[1] == config.edge_dim
    assert torch.isfinite(graph.x).all()


def test_modnet_kan_single_target_forward() -> None:
    model = MODNetKAN(
        n_feat=8,
        targets=[[["band_gap"]]],
        num_neurons=([12], [8], [4], []),
        block_type="kan",
        kan_grid_size=3,
    )
    output = model(torch.randn(5, 8))
    assert output.shape == (5,)
    assert model.target_names == ["band_gap"]
    assert torch.isfinite(output).all()


def test_modnet_kan_multi_target_forward() -> None:
    model = MODNetKAN(
        n_feat=6,
        targets=[[["bulk_modulus"], ["shear_modulus", "poisson_ratio"]]],
        num_neurons=([10], [8], [4], []),
        block_type="mlp",
    )
    output = model(torch.randn(3, 6))
    assert output.shape == (3, 3)
    assert model.target_names == ["bulk_modulus", "shear_modulus", "poisson_ratio"]
    assert torch.isfinite(output).all()


def test_modnet_feature_processor_selects_and_transforms() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "good_a": [0.0, 1.0, 2.0, 3.0],
            "good_b": [3.0, 2.0, 1.0, 0.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
            "with_nan": [1.0, float("nan"), 3.0, 4.0],
            "with_inf": [0.0, 1.0, float("inf"), 3.0],
        }
    )
    targets = [0.0, 1.0, 2.0, 3.0]
    processor = MODNetFeatureProcessor(n_features=3, random_state=0)
    features = processor.fit_transform(frame, targets)

    assert features.shape == (4, 3)
    assert processor.selected_columns_ is not None
    assert "constant" not in processor.selected_columns_
    assert torch.isfinite(torch.from_numpy(features)).all()


def test_modnet_feature_processor_classification_relevance() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "signal": [0.0, 0.1, 0.9, 1.0, 0.2, 0.8],
            "inverse": [1.0, 0.9, 0.1, 0.0, 0.8, 0.2],
            "constant": [5.0] * 6,
        }
    )
    targets = [0, 0, 1, 1, 0, 1]
    processor = MODNetFeatureProcessor(n_features=2, random_state=0, task_type="classification")
    features = processor.fit_transform(frame, targets)

    assert features.shape == (6, 2)
    assert processor.selected_columns_ is not None
    assert "constant" not in processor.selected_columns_
    assert torch.isfinite(torch.from_numpy(features)).all()


def test_pymatgen_feature_frame_modnet_forward() -> None:
    from pymatgen.core import Composition

    frame = make_feature_frame(
        [Composition("NaCl"), Composition("SiO2"), Composition("Al2O3")],
        preset="pymatgen-composition",
    )
    processor = MODNetFeatureProcessor(n_features=8, random_state=1)
    features = processor.fit_transform(frame, [1.0, 2.0, 3.0])
    model = MODNetKAN(
        n_feat=features.shape[1],
        num_neurons=([8], [4], [4], []),
        kan_grid_size=3,
    )
    output = model(torch.from_numpy(features))
    assert output.shape == (3,)
    assert torch.isfinite(output).all()
