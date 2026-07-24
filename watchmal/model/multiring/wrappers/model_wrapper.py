# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
import torch
import torch.nn as nn

class MultiRingModel(nn.Module):
    """
    Forward IO (event-batched sparse):
      batch["coords"]: int32 [N_all, 4] (b,z,y,x)
      batch["feats"]:  float [N_all, C_in]
      batch["meta"]:   List[Dict] length B
    Outputs: dict (head-dependent), always includes "per_event" list outputs.
    """
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        enc = self.encoder(batch)     # standardized dict
        out = self.head(enc, batch)   # standardized dict
        return out
