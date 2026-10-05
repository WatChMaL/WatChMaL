import torch.nn as nn


# ============================================================
# Node encoder
# ============================================================
class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, use_gelu=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU() if use_gelu else nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x):
        return self.net(x)
