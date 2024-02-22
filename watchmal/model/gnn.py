# torch imports
import torch
import torch.nn.functional as F

# pyg imports
import torch_geometric


class GCN(torch.nn.Module):
    def __init__(self, in_feat=8, h_feat=8, num_output_channels=4):
        '''
        Graph Convolutional Network (GCN)
        The Graph Neural Network from the 
        “Semi-supervised Classification with Graph Convolutional Networks” paper, 
        using the GCNConv operator for message passing.
        '''
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = torch_geometric.nn.GCNConv(in_feat, h_feat)
        self.conv2 = torch_geometric.nn.GCNConv(h_feat, h_feat*2)
        self.conv3 = torch_geometric.nn.GCNConv(h_feat*2, h_feat*4)
        self.conv4 = torch_geometric.nn.GCNConv(h_feat*4, h_feat*8)
        self.lin = torch.nn.Linear(h_feat * 8, num_output_channels)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)
        x = x.tanh()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = torch_geometric.nn.global_mean_pool(
            x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class ResGCN(torch.nn.Module):
    def __init__(self,  in_feat=8, h_feat=128, num_classes=4, num_layers=6, dropout=0.1):
        '''
        Residual Graph Convolutional Network (ResGCN)
        The skip connection operations from the 
        “DeepGCNs: Can GCNs Go as Deep as CNNs?” 
        and “All You Need to Train Deeper GCNs” papers.
        The implemented skip connections includes the pre-activation residual connection ("res+"), 
        the residual connection ("res"), the dense connection ("dense") and no connections ("plain").
        '''
        super().__init__()

        self.node_encoder = torch.nn.Linear(in_feat, h_feat)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = torch_geometric.nn.GENConv(
                h_feat, h_feat, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = torch.nn.LayerNorm(h_feat, elementwise_affine=True)
            act = torch.nn.ReLU(inplace=True)

            layer = torch_geometric.nn.DeepGCNLayer(
                conv, norm, act, block='res+', dropout=dropout, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.classifier = torch.nn.Linear(h_feat, num_classes)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.node_encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index)

        x = torch_geometric.nn.global_add_pool(x, batch)
        return self.classifier(x)


class GINModel(torch.nn.Module):
    def __init__(self, in_feat=8, h_feat=128, num_classes=4, dropout=0.5):
        '''
        Graph Isomorphism Network (GIN)
        The Graph Neural Network from the 
        “How Powerful are Graph Neural Networks?” paper, 
        using the GINConv operator for message passing.
        '''
        super().__init__()
        self.gnn = torch_geometric.nn.GIN(
            in_feat, h_feat, 3, dropout=0.5, jk='cat')
        self.classifier = torch_geometric.nn.MLP(
            [h_feat, h_feat*2, num_classes], norm="batch_norm", dropout=dropout)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.gnn(x, edge_index)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.classifier(x)
        return x


class DyEdCNN(torch.nn.Module):
    def __init__(self, in_feat=8, num_classes=4, k=35, dropout_rate=0.1, aggr='max'):
        '''
        Dynamic Edge Convolutional Neural Network (DyEdCNN)
        The Graph Neural Network from the 
        “Dynamic Graph CNN for Learning on Point Clouds” paper,
        using the EdgeConv operator for message passing.
        '''
        super().__init__()

        self.conv1 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP([2 * in_feat, 64, 64, 64]), k, aggr)
        self.conv2 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP([2 * 64, 128]), k, aggr)
        self.lin1 = torch.nn.Linear(128 + 64, 256)

        self.mlp = torch_geometric.nn.MLP(
            [256, 128, 64, num_classes], dropout=dropout_rate, norm="batch_norm")

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = torch_geometric.nn.global_max_pool(out, batch)
        out = self.mlp(out)
        return out
