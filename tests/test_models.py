import torch
from torch_geometric.loader import DataLoader

from cgcnn_pyg_kan.data import SyntheticConfig, make_synthetic_crystal_dataset
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
