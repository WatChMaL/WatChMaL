import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


# ============================================================
# Node encoder
# ============================================================
class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Local PMT GAT (intra-mPMT locality)
# ============================================================
class LocalPMTGAT(nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()
        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.q = nn.Linear(hidden_channels, hidden_channels)
        self.k = nn.Linear(hidden_channels, hidden_channels)
        self.v = nn.Linear(hidden_channels, hidden_channels)

        self.update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index):
        row, col = edge_index

        Q = self.q(x).view(-1, self.num_heads, self.head_dim)
        K = self.k(x).view(-1, self.num_heads, self.head_dim)
        V = self.v(x).view(-1, self.num_heads, self.head_dim)

        scores = (Q[row] * K[col]).sum(-1) / self.head_dim**0.5
        attn = softmax(scores, index=row)

        msg = attn.unsqueeze(-1) * V[col]
        msg = msg.view(-1, x.size(-1))

        agg = torch.zeros_like(x)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        return self.update(torch.cat([x, agg], dim=-1)) + x


# ============================================================
# Inter-mPMT GAT
# ============================================================
class IntermPMTGAT(nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()
        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.q = nn.Linear(hidden_channels, hidden_channels)
        self.k = nn.Linear(hidden_channels, hidden_channels)
        self.v = nn.Linear(hidden_channels, hidden_channels)

        self.update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index):
        row, col = edge_index

        Q = self.q(x).view(-1, self.num_heads, self.head_dim)
        K = self.k(x).view(-1, self.num_heads, self.head_dim)
        V = self.v(x).view(-1, self.num_heads, self.head_dim)

        scores = (Q[row] * K[col]).sum(-1) / self.head_dim**0.5
        attn = softmax(scores, index=row)

        msg = attn.unsqueeze(-1) * V[col]
        msg = msg.view(-1, x.size(-1))

        agg = torch.zeros_like(x)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        return self.update(torch.cat([x, agg], dim=-1)) + x


# ============================================================
# Hierarchical GAT (final)
# ============================================================
class HierarchicalGAT(nn.Module):
    def __init__(
        self,
        pmt_in_channels,
        hidden_channels,
        out_channels,
        num_local_layers=1,
        num_inter_layers=3,
        num_heads_local=1, # number of heads for local, intra-mPMT GAT
        num_heads_global=4, # number of heads for global, inter-mPMT GAT
        dropout=0.0,
        use_nhits=False,
        use_event_total_charge=False,
    ):
        super().__init__()

        self.use_nhits = use_nhits
        self.use_event_total_charge = use_event_total_charge

        self.pmt_encoder = NodeEncoder(pmt_in_channels, hidden_channels)

        self.local_pmt_layers = nn.ModuleList([
            LocalPMTGAT(hidden_channels, num_heads_local)
            for _ in range(num_local_layers)
        ])


        self.inter_layers = nn.ModuleList([
            IntermPMTGAT(hidden_channels, num_heads_global)
            for _ in range(num_inter_layers)
        ])

        extra = int(use_nhits) + int(use_event_total_charge)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels + extra, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, out_channels),
        )

        self.drop = nn.Dropout(p=dropout)

    def forward(self, data):
        
        # Match collate keys: pmt_data (dict with "data" = batch), mpmt_data = batch
        if isinstance(data, dict) and "pmt_data" in data:
            pmt_batch = data["pmt_data"]
            pmt_data = pmt_batch["data"] if isinstance(pmt_batch, dict) else pmt_batch
            mpmt_data = data.get("mpmt_data", None)
        else:
            pmt_data = data
            mpmt_data = None

        # ---- PMT inputs
        x = pmt_data.x[:, :-1]
        tube_ids = pmt_data.x[:, -1].long()
        batch = pmt_data.batch

        # ---- batch-safe mPMT remapping
        unique, pmt_to_mPMT = torch.unique(
            torch.stack([batch, tube_ids], dim=1),
            return_inverse=True,
            dim=0,
        )
        num_mPMTs = unique.size(0)
        mPMT_batch = unique[:, 0]

        # ---- encode PMTs
        x = self.pmt_encoder(x)

        # ---- local PMT graph
        for layer in self.local_pmt_layers:
            x = layer(x, pmt_data.edge_index)

        # aggregate to mPMT level by mean pooling each mPMT's PMTs
        x_mPMT = torch.zeros(num_mPMTs, x.size(1), device=x.device)
        x_mPMT = x_mPMT.index_add_(0, pmt_to_mPMT, x)
        counts = torch.bincount(pmt_to_mPMT, minlength=num_mPMTs).unsqueeze(1).clamp(min=1)
        x_mPMT = x_mPMT / counts

        if mpmt_data is not None and hasattr(mpmt_data, "edge_index"):
            for layer in self.inter_layers:
                x_mPMT = layer(x_mPMT, mpmt_data.edge_index)
                x_mPMT = self.drop(x_mPMT)

        # ---- pool to event
        x_evt = global_mean_pool(x_mPMT, mPMT_batch)

        if self.use_nhits:
            x_evt = torch.cat([x_evt, pmt_data.n_hits.unsqueeze(1)], dim=1)

        if self.use_event_total_charge:
            x_evt = torch.cat(
                [x_evt, pmt_data.event_total_charge.unsqueeze(1)], dim=1
            )

        return self.classifier(x_evt)
