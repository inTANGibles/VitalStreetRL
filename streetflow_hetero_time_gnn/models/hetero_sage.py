"""
Hetero GraphSAGE: 2-layer message passing, regression head per node type.
Input 11 (空间特征，不含时间), output 1 scalar per node (only public has labels; shop head still present).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv, Linear


class HeteroSAGE(nn.Module):
    def __init__(self, in_channels: int = 11, hidden_channels: int = 64, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        node_types = ["shop", "public"]
        edge_types = [
            ("shop", "adj", "shop"),
            ("shop", "adj", "public"),
            ("public", "adj", "shop"),
            ("public", "adj", "public"),
        ]
        self.conv1 = HeteroConv({
            e: SAGEConv((in_channels, in_channels), hidden_channels) for e in edge_types
        }, aggr="sum")
        self.conv2 = HeteroConv({
            e: SAGEConv(hidden_channels, hidden_channels) for e in edge_types
        }, aggr="sum")
        self.lin_shop = Linear(hidden_channels, out_channels)
        self.lin_public = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {"shop": [n_shop, 11], "public": [n_public, 11]}
        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {k: h.relu() for k, h in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)
        h_dict = {k: h.relu() for k, h in h_dict.items()}
        out_shop = self.lin_shop(h_dict["shop"])
        out_public = self.lin_public(h_dict["public"])
        return {"shop": out_shop, "public": out_public}
