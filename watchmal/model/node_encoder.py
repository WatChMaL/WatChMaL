import torch.nn as nn


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
