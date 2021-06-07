import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
#.......................................................................#


def pn_MLP(channels, batch_norm=True):
    return Sequential(*[
        Sequential(Linear(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class dgcnnFeat(nn.Module):
    
    def __init__(self, feature_transform=False, out_channels=2, k=20, aggr='max'):
        super().__init__()
        
        self.feature_transform = feature_transform
        self.conv1 = DynamicEdgeConv(pn_MLP([2*7, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(pn_MLP([2 * 64, 128]), k, aggr)


    def forward(self, data):
        pos, batch = data.x, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = torch.cat([x1, x2], dim=1)
        return [out, batch]