import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax

# Watchmal import
from watchmal.utils.logging_utils_caverns import setup_logging
log = setup_logging(__name__)



class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, max_position=10):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels)
        )
        # Optional positional embedding (e.g. towall bin index or hit time rank)
        # self.pos_embed = nn.Embedding(max_position, hidden_channels)
    
    def forward(self, x, pos_idx=None):

        x = self.mlp(x)
        # if pos_idx is not None:
        #     x += self.pos_embed(pos_idx)
        
        return x

class Concat_and_Wo_MLP_AttentionLayer(nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0

        self.w_q = nn.Linear(hidden_channels, hidden_channels)
        self.w_k = nn.Linear(hidden_channels, hidden_channels)
        self.w_v = nn.Linear(hidden_channels, hidden_channels)

        self.w_o = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        Q = self.w_q(x).view(-1, self.num_heads, self.head_dim)
        K = self.w_k(x).view(-1, self.num_heads, self.head_dim)
        V = self.w_v(x).view(-1, self.num_heads, self.head_dim)

        attn_scores = (Q[row] * K[col]).sum(dim=-1) / self.head_dim**0.5
        attn_weights = softmax(attn_scores, index=row)

        out = attn_weights.unsqueeze(-1) * V[col]
        out = out.view(-1, self.num_heads * self.head_dim)

        agg = torch.zeros_like(x).scatter_add_(0, row.unsqueeze(-1).expand_as(out), out)

        out = torch.cat([x, agg], dim=-1)
        out = self.w_o(out)
        return out + x


class VanillaAttentionLayer(nn.Module):
    def __init__(self, hidden_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0

        self.w_q = nn.Linear(hidden_channels, hidden_channels)
        self.w_k = nn.Linear(hidden_channels, hidden_channels)
        self.w_v = nn.Linear(hidden_channels, hidden_channels)

        self.w_o = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index):
        row, col = edge_index
        Q = self.w_q(x).view(-1, self.num_heads, self.head_dim)
        K = self.w_k(x).view(-1, self.num_heads, self.head_dim)
        V = self.w_v(x).view(-1, self.num_heads, self.head_dim)

        attn_scores = (Q[row] * K[col]).sum(dim=-1) / self.head_dim**0.5
        attn_weights = softmax(attn_scores, index=row)

        out = attn_weights.unsqueeze(-1) * V[col]
        out = out.view(-1, self.num_heads * self.head_dim)

        Z = torch.zeros_like(x).scatter_add_(0, row.unsqueeze(-1).expand_as(out), out)

        out = self.w_o(Z)
        out = self.norm(self.dropout(out) + x)
        return out

ATTENTION_LAYERS = {
    "vanilla": VanillaAttentionLayer,
    "concat_and_wo_mlp": Concat_and_Wo_MLP_AttentionLayer,
}


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, num_heads=4, max_position=10, use_nhits: bool=False, use_event_total_charge: bool=False, attention_type: str='vanilla'):
        super().__init__()

        assert attention_type in ATTENTION_LAYERS, f"Attention type must be one of {list(ATTENTION_LAYERS.keys())}. Got {attention_type}"
        AttentionLayerCls = ATTENTION_LAYERS[attention_type]
        
        self.encoder = NodeEncoder(in_channels, hidden_channels, max_position=max_position)
        self.attn_layers = nn.ModuleList([AttentionLayerCls(hidden_channels, num_heads) for _ in range(num_layers)])
        self.pool = global_mean_pool
        
        self.use_nhits = use_nhits
        self.use_event_total_charge = use_event_total_charge

        first_hidden_channel = hidden_channels + int(use_nhits)
        first_hidden_channel += int(use_event_total_charge)
        
        self.classifier = nn.Sequential(
            nn.Linear(first_hidden_channel, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x =  self.encoder(x)
        for layer in self.attn_layers:
            x = layer(x, edge_index)
        x = self.pool(x, batch)
        
        # Add the n_hits info to the input of the mlp
        if self.use_nhits:
            assert hasattr(data, 'n_hits'), f"Use n_hits set to true but data has not n_hits attributs. Data : {data}"
            nhits = data.n_hits.to(x.dtype).to(x.device).unsqueeze(1)
            x = torch.cat([x, nhits], dim=1) 

        if self.use_event_total_charge:
            assert hasattr(data, 'event_total_charge'), f"Use event_total_charge set to true but data has not event_total_charge attributs. Data : {data}"
            event_total_charge = data.event_total_charge.to(x.dtype).to(x.device).unsqueeze(1)
            x = torch.cat([x, event_total_charge], dim=1) 

        x = self.classifier(x)
        return x