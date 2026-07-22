
import torch
import torch.nn.functional as F

import torch_geometric


class BaseGCN(torch.nn.Module):
    def __init__(self, in_feat=8, h_feat=8, num_classes=4, dropout=0.1):
        '''
        Graph Convolutional Network (GCN)
        The Graph Neural Network from the 
        “Semi-supervised Classification with Graph Convolutional Networks” paper, 
        using the GCNConv operator for message passing.
        '''
        super().__init__()

        self.conv1 = torch_geometric.nn.GCNConv(in_feat, h_feat)
        self.conv2 = torch_geometric.nn.GCNConv(h_feat, h_feat*4)
        self.conv3 = torch_geometric.nn.GCNConv(h_feat*4, h_feat*8)
        self.conv4 = torch_geometric.nn.GCNConv(h_feat*8, h_feat*16)
        self.lin_1 = torch.nn.Linear(h_feat*16, h_feat*8)
        self.lin_2 = torch.nn.Linear(h_feat*8, num_classes)

        self.activation_1 = torch.nn.PReLU()
        self.activation_2 = torch.nn.PReLU()
        self.activation_3 = torch.nn.PReLU()

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.activation_1(x)
        x = self.conv2(x, edge_index)
        x = self.activation_2(x)
        x = self.conv3(x, edge_index)
        x = self.activation_3(x)
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = torch_geometric.nn.global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply linear layers
        x = self.lin_1(x)
        x = self.dropout(x)
        x = self.lin_2(x)

        return x