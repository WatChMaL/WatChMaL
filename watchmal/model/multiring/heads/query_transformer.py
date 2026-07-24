# -*- coding: utf-8 -*-
"""
Query Transformer head for MultiRing segmentation model.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_functions import apply_normalized_activation

class SinPE(nn.Module):
    def __init__(self, bands: int, base: float = 2*math.pi, scale: float = 2.0, d_out: int = 256):
        super().__init__()
        self.omegas = nn.Parameter(torch.tensor([base*(scale**j) for j in range(bands)], dtype=torch.float32), requires_grad=False)
        self.proj = nn.Linear(6*bands, d_out)

    def forward(self, zyx: torch.Tensor, grid: Tuple[int,int,int]) -> torch.Tensor:
        Z,Y,X = [float(g) for g in grid]
        x = zyx[:,2] / max(X-1,1.0)
        y = zyx[:,1] / max(Y-1,1.0)
        z = zyx[:,0] / max(Z-1,1.0)
        xyz = torch.stack([x,y,z], dim=1)  # [U,3]
        phase = xyz.unsqueeze(-1)*self.omegas  # [U,3,B]
        pe = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(1)  # [U,6B]
        return self.proj(pe)

class QueryDecoder(nn.Module):
    def __init__(self, d: int, nhead: int, layers: int, n_queries: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d))
        dec_layer = nn.TransformerDecoderLayer(d_model=d, nhead=nhead, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=layers)

    def forward(self, mem: torch.Tensor, mem_pe: torch.Tensor) -> torch.Tensor:
        src = mem + mem_pe             # [U,D]
        tgt = self.query.unsqueeze(0)  # [1,N,D]
        out = self.dec(tgt, src.unsqueeze(0)).squeeze(0)  # [N,D]
        return out

class QueryPerVoxelSoftmaxHead(nn.Module):
    """
    Outputs per-event:
      Z: [V_b, N+1], Pi: [V_b, N+1], H: [N+1,D]
    """
    def __init__(
        self,
        d_model: int = 256,
        c_vox: int = 64,
        c_mask: int = 96,
        n_queries: int = 2,
        nhead: int = 8,
        layers: int = 3,
        pe_bands: int = 8,
        voxel_attention: bool = False,
        activation: str = "softmax",
        k=None,
        alpha: float = 1.5,
        n_iter: int = 50,
        ensure_sum_one: bool = True,
    ):
        super().__init__()
        self.pe = SinPE(pe_bands, d_out=d_model)

        self.dec = QueryDecoder(d=d_model, nhead=nhead, layers=layers, n_queries=n_queries + 1)

        self.vox_to_mask = nn.Linear(c_vox, c_mask)
        self.q_to_mask = nn.Linear(d_model, c_mask)

        self.n_queries = n_queries + 1

        self.voxel_attention = voxel_attention
        self.vox_to_dec = nn.Linear(c_vox, d_model)

        self.activation = activation
        self.k = k
        self.alpha = alpha
        self.n_iter = n_iter
        self.ensure_sum_one = ensure_sum_one

    def _empty_event(self, vox_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zero placeholders (Z, Pi, H) for an event with no voxels/memory."""
        return (
            vox_f.new_zeros((0, self.n_queries)),
            vox_f.new_zeros((0, self.n_queries)),
            vox_f.new_zeros((self.n_queries, self.q_to_mask.in_features)),
        )

    def forward(self, enc: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        grid = batch["meta"][0]["grid_size"]

        Z_list: List[torch.Tensor] = []
        Pi_list: List[torch.Tensor] = []
        H_list: List[torch.Tensor] = []

        voxel_idx_list = enc.get("voxel_idx_list", None)

        for idx, (mem_f, mem_c, vox_f) in enumerate(
            zip(enc["mem_feat_list"], enc["mem_coord_list"], enc["voxel_feat_list"])
        ):

            if vox_f.numel() == 0:
                z, pi, h = self._empty_event(vox_f)
                Z_list.append(z); Pi_list.append(pi); H_list.append(h)
                continue

            if self.voxel_attention:
                if voxel_idx_list is None:
                    raise KeyError("Expected enc['voxel_idx_list'] when voxel_attention=True.")
                vox_idx = voxel_idx_list[idx]                 # [V,3] (z,y,x)
                mem_for_dec = self.vox_to_dec(vox_f)          # [V,D]
                pe_for_dec = self.pe(vox_idx.float(), grid)   # [V,D]
            else:
                if mem_f.numel() == 0:
                    z, pi, h = self._empty_event(vox_f)
                    Z_list.append(z); Pi_list.append(pi); H_list.append(h)
                    continue
                mem_for_dec = mem_f                           # [U,D]
                pe_for_dec = self.pe(mem_c.float(), grid)     # [U,D]

            H = self.dec(mem_for_dec, pe_for_dec)             # [N+1,D]

            E = self.vox_to_mask(vox_f)                       # [V,Cm]
            P = self.q_to_mask(H)                             # [N+1,Cm]
            Z = E @ P.t() / math.sqrt(E.shape[1])             # [V,N+1]
            Pi = apply_normalized_activation(
                Z,
                activation=self.activation,
                dim=1,
                k=self.k,
                alpha=self.alpha,
                n_iter=self.n_iter,
                ensure_sum_one=self.ensure_sum_one,
                return_support_size=False,
            )

            Z_list.append(Z)
            Pi_list.append(Pi)
            H_list.append(H)

        return {"Z_list": Z_list, "Pi_list": Pi_list, "H_list": H_list}