import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn import GraphNorm
from torch_scatter import scatter, scatter_add, scatter_max


def safe_normalize(x, dim=1, eps=1e-8):
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / (norm + eps)

class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, max_position=10, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
          
        )

    def forward(self, x, pos_idx=None):
        return self.mlp(x)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_channels, num_heads=4, dropout=0.0):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.hidden = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.out_lin = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.mlp_fc1 = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_ln = nn.RMSNorm(hidden_channels)
        self.mlp_fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.mlp_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def aggregate_messages(self, msgs, index, num_nodes):
        return scatter_add(msgs, index, dim=0, dim_size=num_nodes)

    def forward(self, x, edge_index, token_indices=None, return_node_scores=False):
        row, col = edge_index
        N = x.size(0)

        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        q_row = Q[row]
        k_col = K[col]
        v_col = V[col]

        attn_scores = (q_row * k_col).sum(dim=-1) / self.scale
        
        attn_weights = softmax(attn_scores, index=row)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        weighted_v = v_col * attn_weights.unsqueeze(-1)
        agg = self.aggregate_messages(weighted_v.view(-1, self.hidden), row, N)

        residual = x
        x = self.mlp_ln(x)
        hidden = F.relu(self.mlp_fc1(torch.cat([x, agg], dim=-1)))
        hidden = self.mlp_dropout(hidden)
        hidden = self.mlp_fc2(hidden)
        out = residual + self.out_lin(hidden)


        node_scores = None
        if return_node_scores:
            node_scores = scatter_add(attn_weights, col, dim=0, dim_size=N)
            node_scores = node_scores.sum(dim=1)
        if token_indices is not None:
            token_out = out[token_indices]
            if return_node_scores:
                return out, token_out, node_scores
            return out, token_out

        if return_node_scores:
            return out, node_scores

        return out


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 num_layers=4, num_heads=4,
                 use_nhits=False, use_event_total_charge=False, use_vertex=False, use_direction = False,
                 dropout=0.0,reg_dims=[1,4,3],normalize_heads=None):
        super().__init__()
        self.use_vertex = use_vertex
        self.use_direction = use_direction
        self.use_nhits = use_nhits
        self.use_event_total_charge = use_event_total_charge
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.blank_token = nn.Parameter(torch.randn(1, hidden_channels))

        # Compute prefit dimension for projection into token
        prefit_dim = 0
        if use_vertex:
            prefit_dim += 3
        if use_nhits:
            prefit_dim += 1
        if use_event_total_charge:
            prefit_dim += 1
        if use_direction:
            prefit_dim +=3
        #node encoding for token off as seems to affect training stability when using prefit
        self.global_token_proj =   nn.Linear(prefit_dim, hidden_channels) 
        self.encoder = NodeEncoder(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels
                    )
        self.layers = nn.ModuleList([
        nn.ModuleList([
            AttentionLayer(hidden_channels, num_heads=num_heads, dropout=dropout),
            AttentionLayer(hidden_channels, num_heads=num_heads, dropout=dropout),
            AttentionLayer(hidden_channels, num_heads=num_heads, dropout=dropout)
        ])
        for _ in range(num_layers)
        ])
        self.reg_dims = reg_dims
        self.reg_heads = len(reg_dims)
        if normalize_heads is None:
            normalize_heads = [False] * self.reg_heads
        assert len(normalize_heads) == self.reg_heads
        self.normalize_heads = normalize_heads

        
        # MLP for each head
        expanded_dim = hidden_channels
        self.heads = nn.ModuleList([
            nn.Sequential(

                nn.LayerNorm(expanded_dim),
                nn.Linear(hidden_channels, expanded_dim),
                nn.ReLU(inplace=True),
                nn.Linear(expanded_dim, out_dim,bias=True)
            )
            for out_dim in reg_dims
        ])
   
      
        self.log_vars  = nn.Parameter(torch.zeros(self.reg_heads))

    def init_token_from_prefit(self, data, batch, prefit=True):
        device = batch.device
        batch_size = int(batch.max().item()) + 1

        token_feats_list = []

        if prefit:
            if self.use_vertex:
                token_feats_list.append(torch.cat([data.v_x, data.v_y, data.v_z], dim=1))
            if self.use_direction:
                token_feats_list.append(torch.cat([data.d_x, data.d_y, data.d_z], dim=1))
            if self.use_nhits:
                token_feats_list.append(data.n_hits.unsqueeze(1))
            if self.use_event_total_charge:
                token_feats_list.append(data.event_total_charge.unsqueeze(1))

            if len(token_feats_list) == 0:
                token_feats = self.blank_token.expand(batch_size, -1)
            else:
                token_feats = torch.cat(token_feats_list, dim=1).to(device)
                token_feats = self.global_token_proj(token_feats)
        else:
            token_feats = self.blank_token.expand(batch_size, -1)

        return token_feats

    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = x.size(0)
        batch_size = int(batch.max().item()) + 1
        ##encode
        x = self.encoder(
                    x
                )

        ##add token
        blank_token = self.init_token_from_prefit(data, batch, prefit=True)
        token_indices_blank = torch.arange(x.size(0), x.size(0) + batch_size, device=x.device)
        x = torch.cat([x, blank_token], dim=0)

        batch_size = int(batch.max().item()) + 1
        token_indices = torch.arange(batch_size, device=x.device) + num_nodes
        src = torch.arange(num_nodes, device=x.device)
        dst = token_indices[batch]
        edge_index_node_token = torch.stack([src, dst], dim=0)
        edge_index_token_node = torch.stack([dst, src], dim=0)
        ##edge_index_token_node = torch.cat([torch.stack([src, dst], dim=0),torch.stack([dst, src], dim=0)], dim=1)

        # --- Attention layers ---
        for layer_1, layer_2, layer_3 in self.layers:
            # Regular edges
            x, token_out = layer_1(x, edge_index, token_indices_blank)
            # Token edges
            x, token_out = layer_2(x, edge_index_node_token, token_indices_blank)
            x, token_out = layer_3(x, edge_index_token_node, token_indices_blank)
        outputs = []
        for i, head in enumerate(self.heads):
            out = head(token_out)
            if self.normalize_heads[i]:
                out = safe_normalize(out, dim=1)
            outputs.append(out)
        
        # return all head outputs and the log_vars for uncertainty-weighted loss
        return outputs, self.log_vars