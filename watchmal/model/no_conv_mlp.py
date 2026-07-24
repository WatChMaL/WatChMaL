from typing import List

import torch
import torch.nn.functional as F
import numpy as np

import torch_geometric
from torch_geometric.nn import ResGatedGraphConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.norm import BatchNorm, LayerNorm, GraphNorm
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter

from torch.nn import Dropout, Linear, Sigmoid, LogSoftmax, ReLU




class NoConvMLP(torch.nn.Module):
    def __init__(self, num_node_features, max_nodes, hidden_layers):

        super().__init__()
        self.max_nodes = max_nodes
        self.num_node_features = num_node_features
        self.input_dim = max_nodes * num_node_features

        layers = []
        in_dim = self.input_dim
        for h in hidden_layers:
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(torch.nn.ReLU())
            in_dim = h

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, data):
        x, batch = data.x, data.batch
        num_graphs = batch.max().item() + 1
        padded_graphs = []

        for i in range(num_graphs):
            x_i = x[batch == i]  # [num_nodes_i, num_node_features]
            num_nodes = x_i.size(0)

            if num_nodes < self.max_nodes:
                pad = torch.zeros(self.max_nodes - num_nodes, self.num_node_features, device=x.device)
                x_i = torch.cat([x_i, pad], dim=0)
            elif num_nodes > self.max_nodes:
                x_i = x_i[:self.max_nodes]  # truncate

            padded_graphs.append(x_i.flatten())  # [max_nodes * num_node_features]

        out = torch.stack(padded_graphs)  # [batch_size, input_dim]
        return self.mlp(out).squeeze(-1)  # [batch_size]