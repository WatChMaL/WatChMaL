"""
Sparse UNet3D encoder for multiring segmentation.

This script defines:
- BNAct1d: A 1D batch normalization + ReLU activation module.
- SubMBlock: A residual block with two submanifold sparse convolutions.
- Down: A downsampling block with a strided sparse convolution.
- Up: An upsampling block with a sparse inverse convolution and skip connection.
- SparseUNet3D: The main UNet architecture that processes sparse voxel data and produces multi-level features at bottleneck and voxel levels.
The SparseUNet3D takes as input a batch of sparse voxel data (coordinates and features) and produces:
- full: The final sparse tensor after the UNet.
- full_feat_all: The features of the final sparse tensor.
- full_idx_all: The coordinates of the final sparse tensor.
- bottleneck_feat_list: A list of (projected) deepest-level features regrouped by event.
- bottleneck_coord_list: A list of deepest-level coordinates regrouped by event.
- voxel_feat_list: A list of voxel features regrouped by event.
- voxel_idx_list: A list of voxel coordinates regrouped by event.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import os
from spconv.core import ConvAlgo
import spconv.pytorch as spconv
from spconv.pytorch import (
    SparseConvTensor,
    SubMConv3d,
    SparseConv3d,
    SparseInverseConv3d,
)

class BNAct1d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.norm = nn.LayerNorm(c)     
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(x))


class SubMBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, key: str):
        super().__init__()
        self.c1 = SubMConv3d(
            c_in, c_out, 3,
            bias=False,
            indice_key=key,
            algo=ConvAlgo.Native,
        )
        self.n1 = BNAct1d(c_out)
        self.c2 = SubMConv3d(
            c_out, c_out, 3,
            bias=False,
            indice_key=key,
            algo=ConvAlgo.Native,
        )
        self.n2 = BNAct1d(c_out)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        y = self.c1(x)
        y = y.replace_feature(self.n1(y.features))
        y = self.c2(y)
        y = y.replace_feature(self.n2(y.features))
        return y


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int, key: str):
        super().__init__()
        self.conv = SparseConv3d(
            c_in, c_out, 2,
            stride=2,
            bias=False,
            indice_key=key,
            algo=ConvAlgo.Native,   
        )
        self.n = BNAct1d(c_out)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        y = self.conv(x)
        return y.replace_feature(self.n(y.features))


class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, key_down: str, key_up: str):
        super().__init__()
        self.deconv = SparseInverseConv3d(
            c_in, c_out, 2,
            indice_key=key_down,
            bias=False,
            algo=ConvAlgo.Native,    
        )
        self.n = BNAct1d(c_out)
        self.fuse = SubMConv3d(
            c_out + c_skip, c_out, 1,
            bias=False,
            indice_key=key_up,
            algo=ConvAlgo.Native,
        )

    def forward(self, x: SparseConvTensor, skip: SparseConvTensor) -> SparseConvTensor:
        y = self.deconv(x)
        y = y.replace_feature(self.n(y.features))
        y = y.replace_feature(torch.cat([y.features, skip.features], 1))
        return self.fuse(y)


class SparseUNet3D(nn.Module):
    """
    Coords: [N,4]=[b,z,y,x] or [N,5]=[b,t,z,y,x]. For 4D, time is folded into a composite batch.
    Returns:
      - bottleneck_feat_list:  List[[U_b, c_bottleneck]]  (deepest down-conv level, raw)
      - bottleneck_coord_list: List[[U_b, 3]]
      - voxel_feat_list: List[[V_b, c_out]]
      - voxel_idx_list:  List[[V_b, 3]]

    Design note ("memory" -> "bottleneck"): the encoder previously exposed a learned
    projection of an intermediate level chosen by `mem_level` to a fixed width `d_mem`
    ("memory tokens"). That indirection was dropped: the head now consumes the deepest
    down-conv level directly as the decoder source, so the exposed width is exactly
    `c_bottleneck = channels[-1]` (no projection, no extra hyper-parameters). Consequence:
    when a head feeds these features straight to a `d_model` decoder (voxel_attention=False),
    it needs `channels[-1] == d_model` (asserted head-side). `mem_level`/`d_mem`/`d_bottleneck`
    are kept only so older configs still instantiate; they are ignored.
    """
    def __init__(
        self,
        c_in: int = 5,
        c_stem: int = 64,
        channels: Tuple[int, ...] = (96, 160, 256),
        d_bottleneck=None,  # deprecated/ignored; bottleneck width is channels[-1]
        mem_level=None,     # deprecated/ignored (kept so old configs still instantiate)
        d_mem=None,         # deprecated/ignored
    ):
        super().__init__()
        self.channels = tuple(channels)
        self.c_bottleneck = self.channels[-1]  # deepest down-conv width

        self.stem = SubMBlock(c_in, c_stem, key="subm1")

        self.downs, self.blocks = nn.ModuleList(), nn.ModuleList()
        c_prev = c_stem
        for i, c in enumerate(self.channels):
            self.downs.append(Down(c_prev, c, key=f"down{i+1}"))
            self.blocks.append(SubMBlock(c, c, key=f"subm_down{i+1}"))
            c_prev = c

        self.ups, self.upblocks, self.up_skip_levels = nn.ModuleList(), nn.ModuleList(), []
        c_in_up = self.channels[-1]
        L = len(self.channels)
        for d in range(L - 1, -1, -1):
            c_out = (self.channels[d - 1] if d > 0 else c_stem)
            c_skip = (self.channels[d - 1] if d > 0 else c_stem)
            self.ups.append(Up(c_in_up, c_skip, c_out, key_down=f"down{d+1}", key_up=f"up{d+1}"))
            self.upblocks.append(SubMBlock(c_out, c_out, key=f"subm_up{d+1}"))
            self.up_skip_levels.append(d)
            c_in_up = c_out

        self.out_proj = SubMConv3d(c_in_up, c_stem, 1, bias=False, indice_key="subm1")

    @staticmethod
    def _build_sparse(feats: torch.Tensor,
                      coords4: torch.Tensor,
                      grid: Tuple[int, int, int]) -> SparseConvTensor:
        feats = feats.contiguous()
        coords4 = coords4.contiguous().to(feats.device, non_blocking=True).int()
        batch_size = int(coords4[:, 0].max().item()) + 1 if coords4.numel() else 0
        return spconv.SparseConvTensor(
            features=feats,
            indices=coords4,
            spatial_shape=torch.Size(list(grid)),
            batch_size=batch_size,
        )

    @staticmethod
    def _regroup_by_batch(x: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
        """Split rows of x into one tensor per event, keyed by the batch column."""
        if x.numel() == 0:
            return []
        return [x[batch_idx == b] for b in range(batch_size)]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        coords4: torch.Tensor = batch["coords"]
        feats: torch.Tensor = batch["feats"]
        meta: List[Dict[str, Any]] = batch["meta"]
        grid = meta[0]["grid_size"]

        batch_size = int(coords4[:, 0].max().item()) + 1 if coords4.numel() else 0
        x = self._build_sparse(feats, coords4, grid)

        x0 = self.stem(x)
        levels = [x0]
        cur = x0
        for d, blk in zip(self.downs, self.blocks):
            cur = blk(d(cur))
            levels.append(cur)

        bottleneck = levels[-1]
        bn_feat_all = bottleneck.features          # raw deepest-conv features [.,c_bottleneck]
        bn_idx_all = bottleneck.indices

        cur = levels[-1]
        for k, d in enumerate(self.up_skip_levels):
            cur = self.upblocks[k](self.ups[k](cur, levels[d]))

        full = self.out_proj(cur)
        full_idx_all = full.indices
        full_feat_all = full.features

        comp_vox = full_idx_all[:, 0]
        comp_bn = bn_idx_all[:, 0]

        return {
            "full": full,
            "full_feat": full_feat_all,
            "full_idx": full_idx_all[:, 1:],
            "bottleneck_feat_list": self._regroup_by_batch(bn_feat_all, comp_bn, batch_size),
            "bottleneck_coord_list": self._regroup_by_batch(bn_idx_all[:, 1:], comp_bn, batch_size),
            "voxel_feat_list": self._regroup_by_batch(full_feat_all, comp_vox, batch_size),
            "voxel_idx_list": self._regroup_by_batch(full_idx_all[:, 1:], comp_vox, batch_size),
        }
