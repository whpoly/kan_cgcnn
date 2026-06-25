import torch
from torch_geometric.loader import DataLoader

from cgcnn_pyg_kan.data import SyntheticConfig, make_synthetic_crystal_dataset
from cgcnn_pyg_kan.materials import StructureGraphConfig, structure_to_graph
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
            conv_kan_hidden_dim=8,
            conv_kan_grid_size=3,
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
        conv_kan_hidden_dim=4,
        conv_kan_grid_size=3,
    )
    output = model(batch)
    assert output.shape == (1,)
    assert graph.edge_attr.shape[1] == config.edge_dim
