import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
import logging
'''
CheRP: Cherenkov Ring Perceiver
This model is a refinement of the CLSGAT architecture, that also closely resembles a Perceiver style architecture, for regression/classification of single ring events.
It uses torch compilation in the sparse attention core to allow processing of many thousands of PMTs per event, this hopefully works for you!
'''
log = logging.getLogger(__name__)
def activation_stats(x):
    return {
        'frac_dead': (x.abs() < 1e-6).float().mean().item(),
        'mean': x.abs().mean().item(),
    }


# ------------------------
# Node encoder
# ------------------------
class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, log_stats=False):
        if not log_stats:
            return self.mlp(x)
        stats = {}
        for name, layer in self.mlp.named_children():
            x = layer(x)
            if isinstance(layer, (nn.ReLU, nn.GELU)):
                stats[f'encoder_{name}'] = activation_stats(x)
        self.act_stats = stats
        return x


# ------------------------
# Token transformer block
# ------------------------
class TokenTransformerBlock(nn.Module):
    def __init__(self, hidden_channels, num_heads, dropout=0.0, pre_norm=False):
        super().__init__()
        self.pre_norm = pre_norm
        self.attn = nn.MultiheadAttention(
            hidden_channels, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Dropout(p=dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

    def _mlp(self, h, log_stats):
        if not log_stats:
            return self.mlp(h)
        stats = {}
        for name, layer in self.mlp.named_children():
            h = layer(h)
            if isinstance(layer, (nn.ReLU, nn.GELU)):
                stats[f'ffn_{name}'] = activation_stats(h)
        self.act_stats = stats
        return h

    def forward(self, x, log_stats=False):
        if self.pre_norm:
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h)
            x = x + attn_out
            return x + self._mlp(self.norm2(x), log_stats)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self._mlp(x, log_stats))


# ------------------------
# Dense token-attention core (t2n direction only)
# ------------------------
class DenseTokenLayerCore(nn.Module):
    def __init__(self, hidden_channels, num_heads, scale, dropout, pre_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.scale = scale
        self.dropout = dropout
        self.pre_norm = pre_norm

    def forward(self, Q, K, V, query_x, mlp_query, batch,
                mlp_weight1, mlp_bias1, mlp_weight2, mlp_bias2,
                norm_weight, norm_bias):
        # Q: (N, num_heads, head_dim); K, V: (batch_size, num_tokens, num_heads, head_dim)
        K_n = K[batch]   # (N, num_tokens, num_heads, head_dim)
        V_n = V[batch]

        attn_scores  = (Q.unsqueeze(1) * K_n).sum(dim=-1) / self.scale   # (N, num_tokens, num_heads)
        attn_weights = F.softmax(attn_scores, dim=1)

        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        weighted_v = V_n * attn_weights.unsqueeze(-1)   # (N, num_tokens, num_heads, head_dim)
        weighted_v = weighted_v.reshape(weighted_v.size(0), weighted_v.size(1), -1)
        agg = weighted_v.float().sum(dim=1)
        agg = agg.to(query_x.dtype)

        received = agg.abs().sum(dim=-1, keepdim=True) > 0

        combined = torch.cat([mlp_query, agg], dim=-1)
        x_out     = F.linear(combined, mlp_weight1, mlp_bias1)
        x_out     = F.gelu(x_out)

        gelu_frac_dead = (x_out.abs() < 1e-6).float().mean()
        gelu_mean      = x_out.abs().mean()

        x_out = F.linear(x_out, mlp_weight2, mlp_bias2)

        if self.training and self.dropout > 0:
            x_out = F.dropout(x_out, p=self.dropout)

        if self.pre_norm:
            out = torch.where(received, query_x + x_out, query_x)
        else:
            normed = F.layer_norm(query_x + x_out, (query_x.size(-1),), norm_weight, norm_bias)
            out = torch.where(received, normed, query_x)
        return out, gelu_frac_dead, gelu_mean


class DenseTokenAttentionLayer(nn.Module):
    def __init__(self, hidden_channels, num_heads=4, dropout=0.0, pre_norm=False):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.hidden    = hidden_channels
        self.num_heads = num_heads
        self.head_dim  = hidden_channels // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.dropout   = dropout
        self.pre_norm  = pre_norm
        self.core      = torch.compile(
            DenseTokenLayerCore(hidden_channels, num_heads, self.scale, dropout, pre_norm)
        )
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.mlp_l1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.mlp_l2 = nn.Linear(hidden_channels, hidden_channels)
        if pre_norm:
            self.norm_q  = nn.LayerNorm(hidden_channels)
            self.norm_kv = nn.LayerNorm(hidden_channels)
        else:
            self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, query_x, kv_tokens, batch, log_stats=False):
        # query_x: (N, hidden) nodes; kv_tokens: (batch_size, num_tokens, hidden)
        num_query = query_x.size(0)
        batch_size, num_tokens, _ = kv_tokens.shape

        if self.pre_norm:
            mlp_query = self.norm_q(query_x)
            kv_in     = self.norm_kv(kv_tokens)
            norm_weight, norm_bias = None, None
        else:
            mlp_query = query_x
            kv_in     = kv_tokens
            norm_weight, norm_bias = self.norm.weight, self.norm.bias

        Q = self.q_proj(mlp_query).view(num_query, self.num_heads, self.head_dim)
        K = self.k_proj(kv_in).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        V = self.v_proj(kv_in).view(batch_size, num_tokens, self.num_heads, self.head_dim)

        out, gelu_frac_dead, gelu_mean = self.core(
            Q, K, V, query_x, mlp_query, batch,
            self.mlp_l1.weight, self.mlp_l1.bias,
            self.mlp_l2.weight, self.mlp_l2.bias,
            norm_weight, norm_bias,
        )

        if log_stats:
            self.act_stats = {'gelu': {
                'frac_dead': gelu_frac_dead.item(),
                'mean': gelu_mean.item(),
            }}

        return out


# ------------------------
# Compiled attention core
# ------------------------
class CompiledLayerCore(nn.Module):
    def __init__(self, hidden_channels, num_heads, scale, dropout, use_cosine=False, pre_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.scale = scale
        self.dropout = dropout
        self.use_cosine = use_cosine
        self.pre_norm = pre_norm

    def forward(self, Q, K, V, query_x, mlp_query, row, col, num_query,
                mlp_weight1, mlp_bias1, mlp_weight2, mlp_bias2,
                norm_weight, norm_bias):
        q_row = Q[row]
        k_col = K[col]
        v_col = V[col]

        if self.use_cosine:
            attn_weights = (F.normalize(q_row, dim=-1) * F.normalize(k_col, dim=-1)).sum(dim=-1)
        else:
            attn_scores  = (q_row * k_col).sum(dim=-1) / self.scale
            attn_weights = softmax(attn_scores, index=row, num_nodes=num_query)

        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        weighted_v = v_col * attn_weights.unsqueeze(-1)

        agg = torch.zeros((num_query, self.num_heads * self.head_dim),
                          device=query_x.device, dtype=torch.float32)
        agg.index_add_(0, row, weighted_v.view(-1, self.num_heads * self.head_dim).float())
        agg = agg.to(query_x.dtype)

        received = agg.abs().sum(dim=-1, keepdim=True) > 0

        combined = torch.cat([mlp_query, agg], dim=-1)
        x_out     = F.linear(combined, mlp_weight1, mlp_bias1)
        x_out     = F.gelu(x_out)

        gelu_frac_dead = (x_out.abs() < 1e-6).float().mean()
        gelu_mean      = x_out.abs().mean()

        x_out = F.linear(x_out, mlp_weight2, mlp_bias2)

        if self.training and self.dropout > 0:
            x_out = F.dropout(x_out, p=self.dropout)

        if self.pre_norm:
            out = torch.where(received, query_x + x_out, query_x)
        else:
            normed = F.layer_norm(query_x + x_out, (query_x.size(-1),), norm_weight, norm_bias)
            out = torch.where(received, normed, query_x)
        return out, gelu_frac_dead, gelu_mean


# ------------------------
# Attention layer
# ------------------------
class AttentionLayer(nn.Module):
    def __init__(self, hidden_channels, num_heads=4, dropout=0.0, use_cosine=False, pre_norm=False):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.hidden   = hidden_channels
        self.num_heads = num_heads
        self.head_dim  = hidden_channels // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.dropout   = dropout
        self.use_cosine = use_cosine
        self.pre_norm  = pre_norm
        self.core      = torch.compile(
            CompiledLayerCore(hidden_channels, num_heads, self.scale, dropout, use_cosine, pre_norm)
        )
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.mlp_l1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.mlp_l2 = nn.Linear(hidden_channels, hidden_channels)
        if pre_norm:
            self.norm_q  = nn.LayerNorm(hidden_channels)
            self.norm_kv = nn.LayerNorm(hidden_channels)
        else:
            self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, query_x, kv_x, edge_index, log_stats=False):
        row, col = edge_index
        num_query = query_x.size(0)
        num_kv    = kv_x.size(0)

        if self.pre_norm:
            mlp_query = self.norm_q(query_x)
            kv_in     = self.norm_kv(kv_x)
            norm_weight, norm_bias = None, None
        else:
            mlp_query = query_x
            kv_in     = kv_x
            norm_weight, norm_bias = self.norm.weight, self.norm.bias

        Q = self.q_proj(mlp_query).view(num_query, self.num_heads, self.head_dim)
        K = self.k_proj(kv_in).view(num_kv, self.num_heads, self.head_dim)
        V = self.v_proj(kv_in).view(num_kv, self.num_heads, self.head_dim)

        out, gelu_frac_dead, gelu_mean = self.core(
            Q, K, V, query_x, mlp_query, row, col, num_query,
            self.mlp_l1.weight, self.mlp_l1.bias,
            self.mlp_l2.weight, self.mlp_l2.bias,
            norm_weight, norm_bias,
        )

        if log_stats:
            self.act_stats = {'gelu': {
                'frac_dead': gelu_frac_dead.item(),
                'mean': gelu_mean.item(),
            }}

        return out


# ------------------------
# CheRP
# ------------------------
class CheRP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=4,
        num_heads=4,
        num_tokens=6,
        num_end_layers=2,
        token_layers_per_step=1,
        use_nhits=False,
        use_event_total_charge=False,
        dropout=0.0,
        node_dropout=0.0,
        reg_dims=[1, 4, 3],
        normalize_heads=None,
        cosine_attention=False,
        pre_norm=False,
        shared_token_transformer=True,
    ):
        super().__init__()

        self.hidden_channels       = hidden_channels
        self.num_heads             = num_heads
        self.num_tokens            = num_tokens
        self.num_layers            = num_layers
        self.use_nhits             = use_nhits
        self.use_event_total_charge = use_event_total_charge
        self.use_global_token      = use_nhits or use_event_total_charge
        self.multitask             = False
        self.node_dropout          = node_dropout
        self.pre_norm              = pre_norm
        self.shared_token_transformer = shared_token_transformer
        self.log_stats             = True  # set to True to enable activation/token-sim logging

        if isinstance(token_layers_per_step, int):
            self.token_layers_per_step = [token_layers_per_step] * num_layers
        else:
            assert len(token_layers_per_step) == num_layers
            self.token_layers_per_step = list(token_layers_per_step)

        num_frozen = num_tokens - 1 if self.use_global_token else num_tokens
        token_init = torch.empty(num_frozen, hidden_channels)
        nn.init.orthogonal_(token_init)
        self.token_embed = nn.Parameter(token_init)

        if self.use_global_token:
            global_dim = int(use_nhits) + int(use_event_total_charge)
            self.global_token_proj = nn.Linear(global_dim, hidden_channels)

        self.encoder = NodeEncoder(in_channels, hidden_channels)

        self.reg_dims  = reg_dims
        self.reg_heads = len(reg_dims)
        if self.reg_heads > 1:
            self.multitask = True
        if normalize_heads is None:
            normalize_heads = [False] * self.reg_heads
        self.normalize_heads = normalize_heads

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_channels * num_tokens),
                nn.Linear(hidden_channels * num_tokens, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_dim, bias=True),
            )
            for out_dim in reg_dims
        ])

        # log_vars only kept when actually doing multitask uncertainty weighting
        if self.multitask:
            self.log_vars = nn.Parameter(torch.zeros(self.reg_heads))

        self.n2t_layers = nn.ModuleList([
            AttentionLayer(hidden_channels, num_heads, dropout, use_cosine=cosine_attention, pre_norm=pre_norm)
            for _ in range(num_layers)
        ])
        self.t2n_layers = nn.ModuleList([
            DenseTokenAttentionLayer(hidden_channels, num_heads, dropout, pre_norm=pre_norm)
            for _ in range(num_layers - 1)
        ])

        # per token-processing sub-step: either one shared (weight-tied) token
        # transformer reused at every invocation, or one independent set of weights
        # per invocation across the whole per-layer loop
        if shared_token_transformer:
            self.token_transformer = TokenTransformerBlock(hidden_channels, num_heads, dropout, pre_norm=pre_norm)
        else:
            total_token_layers = sum(self.token_layers_per_step)
            self.token_transformers = nn.ModuleList([
                TokenTransformerBlock(hidden_channels, num_heads, dropout, pre_norm=pre_norm)
                for _ in range(total_token_layers)
            ])

        self.transformerend_layers = nn.ModuleList([
            TokenTransformerBlock(hidden_channels, num_heads, dropout, pre_norm=pre_norm)
            for _ in range(num_end_layers)
        ])

    # ------------------------
    # Activation stats (only populated when self.log_stats=True)
    # ------------------------
    @property
    def relu_stats(self):
        if not self.log_stats:
            return {}
        stats = {}
        stats.update(getattr(self.encoder, 'act_stats', {}))
        for i, layer in enumerate(self.n2t_layers):
            for k, v in getattr(layer, 'act_stats', {}).items():
                stats[f'attention_{i}_n2t_{k}'] = v
        for i, layer in enumerate(self.t2n_layers):
            for k, v in getattr(layer, 'act_stats', {}).items():
                stats[f'attention_{i}_t2n_{k}'] = v
        if self.shared_token_transformer:
            for k, v in getattr(self.token_transformer, 'act_stats', {}).items():
                stats[f'token_transformer_{k}'] = v
        else:
            for j, tt in enumerate(self.token_transformers):
                for k, v in getattr(tt, 'act_stats', {}).items():
                    stats[f'token_transformer_{j}_{k}'] = v
        for i, block in enumerate(self.transformerend_layers):
            for k, v in getattr(block, 'act_stats', {}).items():
                stats[f'transformerend_{i}_{k}'] = v
        stats.update(getattr(self, '_head_act_stats', {}))
        return stats

    # ------------------------
    # Token initialisation
    # ------------------------
    def get_tokens(self, data, batch, batch_size):
        base = self.token_embed.unsqueeze(0).expand(batch_size, -1, -1)

        if self.use_global_token:
            feats = []
            if self.use_nhits:
                feats.append(data.n_hits.unsqueeze(1))
            if self.use_event_total_charge:
                feats.append(data.event_total_charge.unsqueeze(1))
            global_input = torch.cat(feats, dim=1).to(batch.device)
            global_tok   = self.global_token_proj(global_input).unsqueeze(1)
            all_tokens   = torch.cat([global_tok, base], dim=1)
        else:
            all_tokens = base

        return all_tokens.reshape(batch_size * self.num_tokens, -1)

    # ------------------------
    # Forward
    # ------------------------
    def forward(self, data):
        x, batch   = data.x, data.batch
        num_nodes  = x.size(0)
        batch_size = data.num_graphs
        ls         = self.log_stats   # local alias avoids repeated attr lookup

        nodes = self.encoder(x, log_stats=ls)

        if self.training and self.node_dropout > 0.0:
            mask = torch.bernoulli(
                torch.full((num_nodes, 1), 1.0 - self.node_dropout, device=nodes.device)
            )
            nodes = nodes * mask

        tokens = self.get_tokens(data, batch, batch_size)

        # nodes and tokens are kept as separate tensors throughout; the sparse
        # n2t edge index pairs every node with all num_tokens of its own graph
        node_local  = torch.arange(num_nodes, device=nodes.device).repeat_interleave(self.num_tokens)
        token_local = (
            batch.unsqueeze(1) * self.num_tokens
            + torch.arange(self.num_tokens, device=nodes.device)
        ).reshape(-1)

        edge_index_node_token = torch.stack([token_local, node_local], dim=0)   # query=token, kv=node

        token_layer_idx = 0
        for i in range(self.num_layers):
            tokens = self.n2t_layers[i](tokens, nodes, edge_index_node_token, log_stats=ls)

            tokens_seq = tokens.view(batch_size, self.num_tokens, -1)
            for _ in range(self.token_layers_per_step[i]):
                if self.shared_token_transformer:
                    tokens_seq = self.token_transformer(tokens_seq, log_stats=ls)
                else:
                    tokens_seq = self.token_transformers[token_layer_idx](tokens_seq, log_stats=ls)
                    token_layer_idx += 1
            tokens = tokens_seq.reshape(-1, self.hidden_channels)

            if i < self.num_layers - 1:
                # every node attends over all num_tokens of its own graph's tokens,
                # so this is a dense gather, not an edge list
                nodes = self.t2n_layers[i](nodes, tokens_seq, batch, log_stats=ls)

        token_out = tokens.view(batch_size, self.num_tokens, -1)

        # Token similarity — only computed when logging
        if ls:
            with torch.no_grad():
                token_norm = F.normalize(token_out, p=2, dim=-1)
                sim = torch.bmm(token_norm, token_norm.transpose(1, 2)).mean(dim=0)
                sim.fill_diagonal_(0)
                self.token_sim        = sim.sum(dim=-1) / (self.num_tokens - 1)
                self.token_sim_matrix = sim

        for layer in self.transformerend_layers:
            token_out = layer(token_out, log_stats=ls)

        token_out = token_out.reshape(batch_size, -1)

        head_act_stats = {}
        outputs = []
        for i, head in enumerate(self.heads):
            if not ls:
                h = head(token_out)
            else:
                h = token_out
                for name, layer in head.named_children():
                    h = layer(h)
                    if isinstance(layer, (nn.ReLU, nn.GELU)):
                        head_act_stats[f'head_{i}_{name}'] = activation_stats(h)
            if self.normalize_heads[i]:
                h = F.normalize(h, p=2, dim=1)
            outputs.append(h)

        if ls:
            self._head_act_stats = head_act_stats

        if self.multitask:
            return outputs, self.log_vars
        else:
            return outputs[0]
