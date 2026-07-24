import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from watchmal.model.node_encoder import NodeEncoder



# ============================================================
# Memory-augmented intra-mPMT aggregation GAT
# ============================================================
class MemoryAugmentedmPMTLayer(nn.Module):
    def __init__(self, hidden_channels, memory_dim, num_heads=4):
        super().__init__()
        assert hidden_channels % num_heads == 0

        self.hidden = hidden_channels
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        # PMT projections
        self.pmt_q = nn.Linear(hidden_channels, hidden_channels)
        self.pmt_k = nn.Linear(hidden_channels, hidden_channels)
        self.pmt_v = nn.Linear(hidden_channels, hidden_channels)

        # Memory projections
        self.mem_q = nn.Linear(hidden_channels, hidden_channels)
        self.mem_k = nn.Linear(hidden_channels, hidden_channels)
        self.mem_v = nn.Linear(hidden_channels, hidden_channels)

        self.memory_init = nn.Parameter(torch.randn(1, hidden_channels))

        self.pmt_update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.mem_update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, memory_dim),
        )

    def forward(self, x, pmt_to_mPMT, num_mPMTs):
        n_pmts = x.size(0)

        # ---- initialize memory nodes
        memory = self.memory_init.expand(num_mPMTs, -1)

        # ---- memory attends to PMTs
        Qm = self.mem_q(memory).view(num_mPMTs, self.num_heads, self.head_dim)
        Kp = self.pmt_k(x).view(n_pmts, self.num_heads, self.head_dim)
        Vp = self.pmt_v(x).view(n_pmts, self.num_heads, self.head_dim)

        Qm_exp = Qm[pmt_to_mPMT]
        scores = (Qm_exp * Kp).sum(-1) / self.head_dim**0.5
        attn = softmax(scores, index=pmt_to_mPMT, num_nodes=num_mPMTs)

        msg = attn.unsqueeze(-1) * Vp
        msg = msg.view(n_pmts, -1)

        mem_agg = scatter_add(msg, pmt_to_mPMT, dim=0, dim_size=num_mPMTs)
        memory_out = self.mem_update(torch.cat([memory, mem_agg], dim=-1))

        # ---- PMTs attend to memory
        Qp = self.pmt_q(x).view(n_pmts, self.num_heads, self.head_dim)
        Km = self.mem_k(memory).view(num_mPMTs, self.num_heads, self.head_dim)
        Vm = self.mem_v(memory).view(num_mPMTs, self.num_heads, self.head_dim)

        Km_exp = Km[pmt_to_mPMT]
        Vm_exp = Vm[pmt_to_mPMT]

        scores_p = (Qp * Km_exp).sum(-1) / self.head_dim**0.5
        attn_p = torch.softmax(scores_p, dim=-1)

        msg_p = attn_p.unsqueeze(-1) * Vm_exp
        msg_p = msg_p.view(n_pmts, -1)

        x_out = self.pmt_update(torch.cat([x, msg_p], dim=-1)) + x

        return x_out, memory_out


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
        memory_dim=16,
        num_local_layers=1,
        num_inter_layers=3,
        num_heads_local=4,
        num_heads_global=4,
        use_nhits=False,
        use_event_total_charge=False,
    ):
        super().__init__()

        self.use_nhits = use_nhits
        self.use_event_total_charge = use_event_total_charge

        self.pmt_encoder = NodeEncoder(pmt_in_channels, hidden_channels)

        self.memory_layers = nn.ModuleList([
            MemoryAugmentedmPMTLayer(
                hidden_channels,
                memory_dim if i == num_local_layers - 1 else hidden_channels,
                num_heads_local,
            )
            for i in range(num_local_layers)
        ])

        self.mPMT_encoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
        )

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

    def forward(self, data):
        # Match collate keys: pmt_data (dict with "data" = batch, "target" = labels), mpmt_data = batch
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


        # ---- intra-mPMT memory aggregation ----------------
        for layer in self.memory_layers:
            x, mPMT_feat = layer(x, pmt_to_mPMT, num_mPMTs)

        # ---- mPMT graph
        x_mPMT = self.mPMT_encoder(mPMT_feat)

        if mpmt_data is not None and hasattr(mpmt_data, "edge_index"):
            for layer in self.inter_layers:
                x_mPMT = layer(x_mPMT, mpmt_data.edge_index)

        # ---- pool to event
        x_evt = global_mean_pool(x_mPMT, mPMT_batch)

        if self.use_nhits:
            x_evt = torch.cat([x_evt, pmt_data.n_hits.unsqueeze(1)], dim=1)

        if self.use_event_total_charge:
            x_evt = torch.cat(
                [x_evt, pmt_data.event_total_charge.unsqueeze(1)], dim=1
            )

        return self.classifier(x_evt)
