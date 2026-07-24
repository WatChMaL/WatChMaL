# -*- coding: utf-8 -*-
"""
Voxelization for the multi-ring sparse CNN, shared by the h5 training dataset
and the reconstruction pipeline. Owns the grid geometry and the feature
normalisation; hits can come from h5, a WCSim file, or memory.
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("SPCONV_ALGO", "native")
from spconv.pytorch.utils import PointToVoxel


@dataclass
class VoxelGridConfig:
    axis_limit: float = 4000.0
    grid_size: int = 60

    @property
    def voxel_size(self) -> float:
        return (2.0 * self.axis_limit) / float(self.grid_size)

    @property
    def origin(self) -> np.ndarray:
        return np.array([-self.axis_limit] * 3, dtype=np.float32)


class Voxelizer:
    def __init__(
        self,
        grid: VoxelGridConfig,
        feat_norm: str = "standard",
        feat_norm_idx: Sequence[int] = (0, 1, 2, 3, 4),
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        fmin: Optional[np.ndarray] = None,
        fmax: Optional[np.ndarray] = None,
    ) -> None:
        self.grid = grid
        self.feat_norm = feat_norm
        self.feat_norm_idx = list(feat_norm_idx)
        self._mean, self._std = mean, std
        self._fmin, self._fmax = fmin, fmax

    def voxel_centers(self, indices_zyx: np.ndarray) -> np.ndarray:
        v = self.grid.voxel_size
        z = indices_zyx[:, 0].astype(np.float32)
        y = indices_zyx[:, 1].astype(np.float32)
        x = indices_zyx[:, 2].astype(np.float32)
        xyz = np.stack([x, y, z], axis=1)
        return (self.grid.origin + (xyz + 0.5) * v).astype(np.float32)

    def apply_norm(self, feats: np.ndarray) -> np.ndarray:
        if self._mean is None and self._fmin is None:
            return feats
        cols = self.feat_norm_idx
        x = feats[:, cols]
        if self.feat_norm == "standard":
            x = (x - self._mean) / np.clip(self._std, 1e-6, None)
        elif self.feat_norm == "minmax":
            x = (x - self._fmin) / np.clip(self._fmax - self._fmin, 1e-6, None)
        feats[:, cols] = x
        return feats

    def set_stats(self, *, mean=None, std=None, fmin=None, fmax=None) -> None:
        self._mean, self._std, self._fmin, self._fmax = mean, std, fmin, fmax

    def voxelize(
        self,
        xyz: np.ndarray,
        q: np.ndarray,
        t: np.ndarray,
        parent_ids: Optional[np.ndarray] = None,
        tube_ids: Optional[np.ndarray] = None,
        pe_counts: Optional[np.ndarray] = None,
        max_points_per_voxel: int = 64,
        max_voxels: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        feat_list = [
            xyz.astype(np.float32),
            q.reshape(-1, 1).astype(np.float32),
            t.reshape(-1, 1).astype(np.float32),
        ]
        next_idx = 5

        n_groups = 0
        idx_pid = None
        if parent_ids is not None and parent_ids.size > 0:
            parent_ids = parent_ids.astype(np.int64, copy=False)
            n_groups = int(parent_ids.max()) + 1
            feat_list.append(parent_ids.reshape(-1, 1).astype(np.float32))
            idx_pid = next_idx
            next_idx += 1

        have_map = (tube_ids is not None) and (tube_ids.size == xyz.shape[0])
        idx_tube = None
        if have_map:
            feat_list.append(tube_ids.reshape(-1, 1).astype(np.float32))
            idx_tube = next_idx
            next_idx += 1

        have_pe = (pe_counts is not None) and (pe_counts.size == xyz.shape[0])
        idx_pe = None
        if have_pe:
            feat_list.append(pe_counts.reshape(-1, 1).astype(np.float32))
            idx_pe = next_idx
            next_idx += 1

        points = np.concatenate(feat_list, axis=1)
        v = float(self.grid.voxel_size)
        L = float(self.grid.axis_limit)
        vxl = PointToVoxel(
            vsize_xyz=[v, v, v],
            coors_range_xyz=[-L, -L, -L, L, L, L],
            num_point_features=points.shape[1],
            max_num_points_per_voxel=int(max_points_per_voxel),
            max_num_voxels=int(
                max_voxels if max_voxels is not None
                else max(1, min(points.shape[0], self.grid.grid_size ** 3))
            ),
        )

        voxels, coors, num_points = vxl(torch.from_numpy(points.astype(np.float32)))
        M, T, _ = voxels.shape

        if M == 0:
            out = {
                "coords_zyx": np.zeros((0, 3), dtype=np.int32),
                "voxel_parent_id": np.zeros((0,), dtype=np.int64),
                "voxel_parent_frac": np.zeros((0, n_groups), dtype=np.float32),
                "voxel_parent_num_groups": np.array(n_groups, dtype=np.int64),
                "voxel_num_points": np.zeros((0,), dtype=np.int32),
            }
            if have_map:
                out["voxel_best_tube"] = np.zeros((0,), dtype=np.int32)
            if have_pe:
                out["voxel_n_pe_sum"] = np.zeros((0,), dtype=np.float32)
                out["voxel_parent_npe"] = np.zeros((0, n_groups), dtype=np.float32)
            return np.zeros((0, 2), dtype=np.float32), out

        mask = (torch.arange(T, device=voxels.device)[None, :] < num_points[:, None])

        q_vals = voxels[..., 3]
        t_vals = voxels[..., 4]
        q_sums = (q_vals * mask.to(q_vals.dtype)).sum(dim=1)
        counts = num_points.to(t_vals.dtype).clamp_min_(1)
        t_mean = ((t_vals * mask.to(t_vals.dtype)).sum(dim=1) / counts).unsqueeze(1)
        feats_out = torch.cat([q_sums.unsqueeze(1), t_mean], dim=1).cpu().numpy().astype(np.float32)
        coords_zyx = coors.int().cpu().numpy().astype(np.int32)

        voxel_pid = np.zeros((M,), dtype=np.int64)
        frac = np.zeros((M, n_groups), dtype=np.float32)
        want_npe = have_pe and (idx_pe is not None)
        npe = np.zeros((M, n_groups), dtype=np.float32)
        if idx_pid is not None and n_groups > 0:
            pid_np = voxels[..., idx_pid].cpu().numpy().astype(np.int64)
            q_np = q_vals.cpu().numpy()
            mask_np = mask.cpu().numpy()
            pe_pts_np = voxels[..., idx_pe].cpu().numpy() if want_npe else None
            for vi in range(M):
                m_vi = mask_np[vi]
                p = pid_np[vi][m_vi]
                qv = q_np[vi][m_vi]
                tot = float(qv.sum())
                if tot > 0.0:
                    bc = np.bincount(p, weights=qv, minlength=n_groups)
                    frac[vi, :] = (bc / tot).astype(np.float32)
                    voxel_pid[vi] = int(np.argmax(bc))
                if want_npe and p.size > 0:
                    npe[vi, :] = np.bincount(
                        p, weights=pe_pts_np[vi][m_vi], minlength=n_groups
                    ).astype(np.float32)

        extra: Dict[str, np.ndarray] = {
            "coords_zyx": coords_zyx,
            "voxel_parent_id": voxel_pid,
            "voxel_parent_frac": frac,
            "voxel_parent_num_groups": np.array(n_groups, dtype=np.int64),
            "voxel_num_points": num_points.cpu().numpy().astype(np.int32, copy=False),
        }

        if have_map and idx_tube is not None:
            tid = voxels[..., idx_tube].masked_fill(~mask, -1)
            bestj = torch.argmax(q_vals.masked_fill(~mask, float("-inf")), dim=1)
            extra["voxel_best_tube"] = tid[
                torch.arange(M, device=voxels.device), bestj
            ].to(torch.int32).cpu().numpy()

        if have_pe and idx_pe is not None:
            pe_vals = voxels[..., idx_pe]
            pe_sums = (pe_vals * mask.to(pe_vals.dtype)).sum(dim=1)
            extra["voxel_n_pe_sum"] = pe_sums.cpu().numpy().astype(np.float32, copy=False)
            extra["voxel_parent_npe"] = npe

        return feats_out, extra

    def build_model_input(
        self,
        xyz: np.ndarray,
        q: np.ndarray,
        t: np.ndarray,
        tube_ids: Optional[np.ndarray] = None,
        min_charge: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Hits -> (coords, feats, extra) ready for MultiRingModel, no truth
        involved. extra keeps the raw per-voxel charge/time as voxel_q/voxel_t.
        """
        keep = q >= float(min_charge)
        xyz, q, t = xyz[keep], q[keep], t[keep]
        tube_ids = tube_ids[keep] if tube_ids is not None else None

        feats2, extra = self.voxelize(xyz, q, t, tube_ids=tube_ids)
        coords_zyx = extra["coords_zyx"]
        if coords_zyx.shape[0] == 0:
            return np.zeros((0, 4), np.int32), np.zeros((0, 5), np.float32), extra

        extra["voxel_q"] = feats2[:, 0].copy()
        extra["voxel_t"] = feats2[:, 1].copy()

        centers = self.voxel_centers(coords_zyx)
        feats = np.concatenate(
            [feats2[:, 0:1], centers, feats2[:, 1:2]], axis=1
        ).astype(np.float32)
        feats = self.apply_norm(feats)
        coords = np.concatenate(
            [np.zeros((coords_zyx.shape[0], 1), np.int32), coords_zyx], axis=1
        ).astype(np.int32, copy=False)
        return coords, feats, extra

    @staticmethod
    def stats_signature(params: Dict[str, Any], grid: VoxelGridConfig) -> str:
        """Must stay identical to HyperKSparseCNN3D._stats_signature."""
        return "|".join([
            f"num_batches={params.get('num_batches')}",
            f"seed={params.get('seed')}",
            f"stats_max_events={params.get('stats_max_events')}",
            f"axis={grid.axis_limit}",
            f"grid={grid.grid_size}",
            f"minq={params.get('min_charge')}",
        ])

    @classmethod
    def from_train_output(cls, train_output_dir: str,
                          stats_cache_path: Optional[str] = None) -> "Voxelizer":
        """
        Rebuild the voxelizer a trained model expects: same grid, same feature
        normalisation statistics, loaded from the cache the training run wrote.
        """
        from watchmal.utils.multiring_sparse_helpers import load_train_cfg, to_container

        cfg = load_train_cfg(train_output_dir)
        params = to_container(cfg.data.dataset.params) or {}
        grid_params = dict(params.get("grid", {}) or {})
        grid_params.pop("time_bin_ns", None)
        grid = VoxelGridConfig(**grid_params)

        vx = cls(
            grid=grid,
            feat_norm=params.get("feat_norm", "standard"),
            feat_norm_idx=params.get("feat_norm_indices", (0, 1, 2, 3, 4)),
        )

        if stats_cache_path is None:
            stats_cache_path = params.get("stats_cache_path")
        if stats_cache_path is None:
            sig = cls.stats_signature(params, grid)
            h = hashlib.md5(sig.encode("utf-8")).hexdigest()[:12]
            stats_cache_path = os.path.join(
                str(params.get("base_dir", "")), "precomputed_features",
                f"feature_stats_{h}.npz",
            )

        if not os.path.isfile(stats_cache_path):
            raise FileNotFoundError(
                f"feature-stats cache not found: {stats_cache_path}; inference "
                "must reuse the training statistics, pass stats_cache_path"
            )
        s = np.load(stats_cache_path, allow_pickle=False)
        vx.set_stats(mean=s["mean"].astype(np.float32), std=s["std"].astype(np.float32),
                     fmin=s["fmin"].astype(np.float32), fmax=s["fmax"].astype(np.float32))
        return vx
