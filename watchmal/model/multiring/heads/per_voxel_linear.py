# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerVoxelLinearHead(nn.Module):
    """
    Minimal per-voxel classifier that matches the loss API.

    Returns:
      {
        "Pi_list":      List[Tensor[V_b, Nring+1]]  # probs; col0 = background, cols 1..Nring = ring slots
        "logits_list":  List[Tensor[V_b, Nring+1]]  # raw logits 
      }
    """
    def __init__(self, c_vox: int = 64, c_out: int = 3):
        super().__init__()
        # c_out = Nring+1
        self.c_out = c_out
        self.cls = nn.Linear(c_vox, c_out)

    def forward(self, enc: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        logits_list: List[torch.Tensor] = []
        Pi_list: List[torch.Tensor] = []

        for vox_f in enc["voxel_feat_list"]:
            z = self.cls(vox_f)                 
            p = F.softmax(z, dim=1)           
            logits_list.append(z)
            Pi_list.append(p)

        return {"Pi_list": Pi_list, "logits_list": logits_list}
