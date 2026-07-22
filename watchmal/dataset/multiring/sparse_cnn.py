# -*- coding: utf-8 -*-
"""
HyperKSparseCNN3D Sparse Dataset (HDF5 -> spconv sparse 3D voxelized dataset)
"""

from __future__ import annotations
import os, glob, math, hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

os.environ["SPCONV_ALGO"] = "native"
from spconv.pytorch.utils import PointToVoxel

log = logging.getLogger("watchmal.dataset.sparse_cnn")
log.setLevel(logging.INFO)


@dataclass
class VoxelGridConfig:
    axis_limit: float = 4000.0
    grid_size: int = 60

    @property
    def voxel_size(self) -> float:
        return (2.0 * self.axis_limit) / float(self.grid_size)

    @property
    def origin(self) -> np.ndarray:
        return np.array([-self.axis_limit, -self.axis_limit, -self.axis_limit], dtype=np.float32)


def _rotate_xy_inplace(xyz: np.ndarray, angle: float):
    c, s = math.cos(angle), math.sin(angle)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    xyz[:, 0] = c * x - s * y
    xyz[:, 1] = s * x + c * y


def _rotate_xy_vec3(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    out = np.asarray(v, dtype=np.float32).copy()
    x, y = float(out[0]), float(out[1])
    out[0] = c * x - s * y
    out[1] = s * x + c * y
    return out

def _has_field(arr, name: str) -> bool:
    return (arr is not None) and (arr.dtype is not None) and (arr.dtype.names is not None) and (name in arr.dtype.names)

class _BaseSparseDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        min_charge: float = 0.01,
        grid: Optional[VoxelGridConfig] = None,
        num_batches: Optional[int] = None,
        seed: Optional[int] = None,
        feat_norm: str = "none",
        feat_norm_indices: Optional[List[int]] = None,
        stats_max_events: Optional[int] = None,
        max_parents: int = 3,
        cache_in_ram: bool = False,
        keep_in_digit: bool = True,
        augmentation: bool = True,
        add_meta: bool = False,
        stats_cache_path: Optional[str] = None,
        max_open_files_per_worker: int = 8,
        file_name_pattern: Optional[str] = "wcsim_output_multihit_with_digi_hit.h5",
        **_: Any,
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.min_charge = float(min_charge)
        log.info(f"Initializing {self.__class__.__name__} with base_dir={base_dir}, min_charge={min_charge}, grid={grid}, num_batches={num_batches}, seed={seed}, feat_norm={feat_norm}, feat_norm_indices={feat_norm_indices}, stats_max_events={stats_max_events}, max_parents={max_parents}, cache_in_ram={cache_in_ram}, keep_in_digit={keep_in_digit}, augmentation={augmentation}, add_meta={add_meta}, stats_cache_path={stats_cache_path}, max_open_files_per_worker={max_open_files_per_worker}, file_name_pattern={file_name_pattern}")
        self.grid = grid or VoxelGridConfig()
        self.max_parents = int(max_parents)

        self.num_batches = None if num_batches is None else int(num_batches)
        self.seed = None if seed is None else int(seed)
        self.stats_max_events = None if stats_max_events is None else int(stats_max_events)
        self.file_name_pattern = str(file_name_pattern) if file_name_pattern is not None else "wcsim_output_multihit_with_digi_hit.h5"

        self.rng = np.random.RandomState(self.seed if self.seed is not None else 0)
        self.cache_in_ram = bool(cache_in_ram)
        self._cache: Dict[int, Dict[str, Any]] = {}
        self.keep_in_digit = bool(keep_in_digit)
        self.augmentation = bool(augmentation)
        self.add_meta = bool(add_meta)
        self.max_open_files_per_worker = int(max(1, max_open_files_per_worker))

        files = sorted(
            glob.glob(
                os.path.join(self.base_dir, "**", self.file_name_pattern),
                recursive=True,
            )
        )
        if self.num_batches is not None and len(files) > 0:
            files = files[: max(1, int(self.num_batches))]

        self.files: List[str] = []
        self._index: List[Tuple[int, int]] = []
        self._events_per_file: Dict[str, int] = {}
        self._worker_files: Dict[int, OrderedDict[int, h5py.File]] = {}

        for path in files:
            with h5py.File(path, "r") as f:
                E = f["/events"]
                n_evt = int(E["event_id"].shape[0])

            self.files.append(path)
            self._events_per_file[path] = n_evt

            fidx = len(self.files) - 1
            self._index.extend((fidx, e) for e in range(n_evt))

        if len(self._index) == 0:
            raise RuntimeError(f"No events found under {base_dir}, found {len(files)} files.")

        self.feat_norm = feat_norm
        self.feat_norm_idx = list(feat_norm_indices or [])
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._fmin: Optional[np.ndarray] = None
        self._fmax: Optional[np.ndarray] = None

        if stats_cache_path is None and self.feat_norm != "none" and len(self.feat_norm_idx) > 0:
            sig = self._stats_signature()
            h = hashlib.md5(sig.encode("utf-8")).hexdigest()[:12]
            pre_dir = os.path.join(self.base_dir, "precomputed_features")
            stats_cache_path = os.path.join(pre_dir, f"feature_stats_{h}.npz")
        self._stats_cache_path = stats_cache_path

        if self.feat_norm != "none" and len(self.feat_norm_idx) > 0:
            if not self._try_load_feature_stats():
                self._init_feature_stats(self.stats_max_events)
                self._try_save_feature_stats()

        log.info(
            f"Loaded {self.__class__.__name__} with {len(self)} events from {len(self.files)} files. "
            f"(max_open_files_per_worker={self.max_open_files_per_worker})"
        )

    def __len__(self) -> int:
        return len(self._index)

    def clear_cache(self):
        self._cache.clear()

    def events_per_h5(self) -> Dict[str, int]:
        return dict(self._events_per_file)

    def _get_h5(self, fidx: int) -> h5py.File:
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id

        lru = self._worker_files.get(wid)
        if lru is None:
            lru = OrderedDict()
            self._worker_files[wid] = lru

        f = lru.get(fidx)
        if f is not None:
            lru.move_to_end(fidx, last=True)
            return f

        f = h5py.File(self.files[fidx], "r")
        lru[fidx] = f
        lru.move_to_end(fidx, last=True)

        while len(lru) > self.max_open_files_per_worker:
            _, old_f = lru.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass

        return f

    def close(self) -> None:
        for _, lru in list(self._worker_files.items()):
            for _, h5 in list(lru.items()):
                h5.close()
            lru.clear()
        self._worker_files.clear()

    def _read_event(self, f: h5py.File, eidx: int):
        E = f["/events"]
        h0, h1 = int(E["hit_index"][eidx]), int(E["hit_index"][eidx + 1])
        p0, p1 = int(E["particle_index"][eidx]), int(E["particle_index"][eidx + 1])
        return E["hits"][h0:h1], E["particles"][p0:p1]

    def _read_digi_event(self, f: h5py.File, eidx: int):
        E = f["/events"]
        d0, d1 = int(E["digi_hit_index"][eidx]), int(E["digi_hit_index"][eidx + 1])
        return E["digi_hits"][d0:d1]

    def _read_digi_photon_lists_event(self, f: h5py.File, eidx: int) -> Tuple[np.ndarray, np.ndarray]:
        E = f["/events"]
        d0, d1 = int(E["digi_hit_index"][eidx]), int(E["digi_hit_index"][eidx + 1])
        iptr = E["digi_photon_index"]
        pflat = E["digi_photon_ids"]
        p0, p1 = int(iptr[d0]), int(iptr[d1])
        return (
            pflat[p0:p1].astype(np.int64, copy=False),
            (iptr[d0:d1 + 1] - p0).astype(np.int64, copy=False),
        )

    def _read_trigger_event(self, f: h5py.File, eidx: int) -> Dict[str, Any]:
        E = f["/events"]

        iptr = E["trigger_info_index"]
        vals = E["trigger_info_values"]
        t0, t1 = int(iptr[eidx]), int(iptr[eidx + 1])

        return {
            "n_triggers": int(E["n_triggers"][eidx]),
            "trigger_type": int(E["trigger_type"][eidx]),
            "trigger_time_ns": np.float32(E["trigger_time_ns"][eidx]),
            "trigger_shift_ns": np.float32(E["trigger_shift_ns"][eidx]),
            "trigger_info": vals[t0:t1].astype(np.float32, copy=False),
            "header_event_number": int(E["header_event_number"][eidx]),
            "header_run_number": int(E["header_run_number"][eidx]),
            "header_date_ns": np.int64(E["header_date_ns"][eidx]),
            "header_subevent_number": int(E["header_subevent_number"][eidx]),
        }

    @staticmethod
    def _compute_primary_groups(parts_evt) -> Tuple[List[int], Dict[int, int]]:
        if parts_evt is None or len(parts_evt) == 0:
            return [], {}

        names = parts_evt.dtype.names or ()
        tid = parts_evt["track_id"].astype(int) if "track_id" in names else np.zeros((len(parts_evt),), dtype=int)
        pid = parts_evt["parent_id"].astype(int) if "parent_id" in names else np.zeros((len(parts_evt),), dtype=int)
        pdg = parts_evt["pdg"].astype(int) if "pdg" in names else np.zeros((len(parts_evt),), dtype=int)
        ene = parts_evt["energy_mev"].astype(float) if "energy_mev" in names else np.zeros((len(parts_evt),), dtype=float)

        have_dir = {"dir_x", "dir_y", "dir_z"}.issubset(names)
        if have_dir:
            dxyz = np.stack([parts_evt["dir_x"], parts_evt["dir_y"], parts_evt["dir_z"]], axis=1).astype(float)
        else:
            dxyz = np.zeros((len(parts_evt), 3), dtype=float)

        info: Dict[int, Tuple[int, int, float, np.ndarray]] = {
            int(t): (int(p), int(g), float(e), dxyz[i])
            for i, (t, p, g, e) in enumerate(zip(tid, pid, pdg, ene))
        }

        mask = (pid == 0) & (pdg != 0) & (ene > 0)
        prim_ids = tid[mask].tolist()

        if have_dir and len(prim_ids) > 1:
            dirs = np.array([info[t][3] for t in prim_ids], dtype=float)
            n = np.linalg.norm(dirs, axis=1)
            n[n == 0] = 1.0
            dirs = dirs / n[:, None]
            used = np.zeros(len(dirs), dtype=bool)
            keep = []
            cos_tol = np.cos(np.radians(0.2))
            for i in range(len(dirs)):
                if used[i]:
                    continue
                keep.append(i)
                used |= (dirs @ dirs[i]) >= cos_tol
            prim_ids = [prim_ids[i] for i in keep]

        cache: Dict[int, int] = {}

        def ancestor(t: int) -> int:
            if t in cache:
                return cache[t]
            seen = set()
            cur = t
            while True:
                rec = info.get(cur)
                if rec is None or cur in seen:
                    cache[t] = -1
                    return -1
                p, g, e, _ = rec
                if p == 0 and g != 0 and e > 0:
                    cache[t] = cur
                    return cur
                seen.add(cur)
                cur = p

        return prim_ids, {int(tt): ancestor(int(tt)) for tt in tid}

    def _assign_primary_hits(
        self,
        hits,
        parts,
        q: np.ndarray,
        topk: Optional[int] = None,
        return_info: bool = True,
    ) -> Tuple[np.ndarray, List[int], List[int], List[Dict[str, Any]]]:
        if topk is None:
            topk = max(0, int(getattr(self, "max_parents", 2)))
        else:
            topk = max(0, int(topk))

        _, tid2prim = self._compute_primary_groups(parts)

        if _has_field(hits, "primary_track_id"):
            hit_primary_tid = hits["primary_track_id"].astype(int)
        else:
            hit_tid = hits["track_id"].astype(int) if _has_field(hits, "track_id") else np.full((hits.shape[0],), -1, dtype=int)
            hit_primary_tid = np.array([tid2prim.get(int(t), -1) for t in hit_tid], dtype=int)

        if hit_primary_tid.size == 0 or topk <= 0:
            return np.zeros((hits.shape[0],), dtype=np.int64), [], [0] * topk, []

        u, inv = np.unique(hit_primary_tid, return_inverse=True)
        sums = np.zeros(u.shape[0], dtype=np.float64)
        for k in range(u.shape[0]):
            sums[k] = float(q[inv == k].sum()) if q.size else 0.0

        valid = u != -1
        u_valid = u[valid]
        sums_valid = sums[valid]
        if u_valid.size == 0:
            return np.zeros((hits.shape[0],), dtype=np.int64), [], [0] * topk, []

        order = np.argsort(-sums_valid)
        top = u_valid[order][:topk]
        primary_slots = {int(tid): (i + 1) for i, tid in enumerate(list(top))}
        pid_hits = np.array([primary_slots.get(int(tp), 0) for tp in hit_primary_tid], dtype=np.int64)

        if not return_info:
            return pid_hits, [], [0] * topk, []

        pdg_arr = parts["pdg"].astype(int) if _has_field(parts, "pdg") else np.zeros((len(parts),), dtype=int)
        track_arr = parts["track_id"].astype(int) if _has_field(parts, "track_id") else np.zeros((len(parts),), dtype=int)
        energy_arr = parts["energy_mev"].astype(float) if _has_field(parts, "energy_mev") else np.zeros((len(parts),), dtype=float)
        start_t_arr = parts["start_t_ns"].astype(float) if _has_field(parts, "start_t_ns") else np.zeros((len(parts),), dtype=float)
        dir_x = parts["dir_x"].astype(float) if _has_field(parts, "dir_x") else np.zeros((len(parts),), dtype=float)
        dir_y = parts["dir_y"].astype(float) if _has_field(parts, "dir_y") else np.zeros((len(parts),), dtype=float)
        dir_z = parts["dir_z"].astype(float) if _has_field(parts, "dir_z") else np.zeros((len(parts),), dtype=float)

        pdg_map = {int(t): int(p) for t, p in zip(track_arr, pdg_arr)}
        energy_map = {int(t): float(e) for t, e in zip(track_arr, energy_arr)}
        start_t_map = {int(t): float(tt) for t, tt in zip(track_arr, start_t_arr)}
        dir_map = {int(t): (float(dx), float(dy), float(dz)) for t, dx, dy, dz in zip(track_arr, dir_x, dir_y, dir_z)}

        top_track_ids = [int(t) for t in top]
        top_pdgs = [int(pdg_map.get(int(t), 0)) for t in top_track_ids]
        if len(top_pdgs) < topk:
            top_pdgs = top_pdgs + [0] * (topk - len(top_pdgs))

        top_info: List[Dict[str, Any]] = []
        for tid in top_track_ids:
            pdg_val = int(pdg_map.get(tid, 0))
            ene = float(energy_map.get(tid, 0.0))
            start_t = float(start_t_map.get(tid, 0.0))
            dx, dy, dz = dir_map.get(tid, (0.0, 0.0, 0.0))
            mag = (dx * dx + dy * dy + dz * dz) ** 0.5
            if mag > 0:
                dx /= mag
                dy /= mag
                dz /= mag
            top_info.append({
                "track_id": tid,
                "pdg": pdg_val,
                "energy_mev": ene,
                "time_ns": start_t,
                "dir": (dx, dy, dz),
            })

        return pid_hits, top_track_ids, top_pdgs[:topk], top_info

    @staticmethod
    def _assign_primary_digis(
        digi_q: np.ndarray,
        photon_ids_flat: np.ndarray,
        photon_ptr: np.ndarray,
        hit_primary_slots: np.ndarray,
    ) -> np.ndarray:
        n_digi = int(digi_q.shape[0])
        pid_digi = np.zeros((n_digi,), dtype=np.int64)
        if n_digi == 0 or photon_ptr.size != n_digi + 1:
            return pid_digi

        n_hits = int(hit_primary_slots.shape[0])
        for di in range(n_digi):
            p0, p1 = int(photon_ptr[di]), int(photon_ptr[di + 1])
            if p1 <= p0:
                continue
            pids = photon_ids_flat[p0:p1]
            good = (pids >= 0) & (pids < n_hits)
            if not np.any(good):
                continue
            slots = hit_primary_slots[pids[good]]
            if slots.size == 0:
                continue
            bc = np.bincount(slots.astype(np.int64), minlength=int(hit_primary_slots.max()) + 1)
            pid_digi[di] = int(np.argmax(bc))
        return pid_digi

    def _voxel_centers(self, indices_zyx: np.ndarray) -> np.ndarray:
        v = self.grid.voxel_size
        z = indices_zyx[:, 0].astype(np.float32)
        y = indices_zyx[:, 1].astype(np.float32)
        x = indices_zyx[:, 2].astype(np.float32)
        xyz = np.stack([x, y, z], axis=1)
        return (self.grid.origin + (xyz + 0.5) * v).astype(np.float32)

    def _voxelize_spconv_3d(
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

        idx_q = 3
        idx_t = 4
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
            max_num_voxels=int(max_voxels if max_voxels is not None else max(1, min(points.shape[0], self.grid.grid_size ** 3))),
        )

        pt = torch.from_numpy(points.astype(np.float32))
        voxels, coors, num_points = vxl(pt)
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
        if idx_pid is not None and n_groups > 0:
            pid_vals = voxels[..., idx_pid]
            q_np = q_vals.cpu().numpy()
            pid_np = pid_vals.cpu().numpy().astype(np.int64)
            mask_np = mask.cpu().numpy()
            for vi in range(M):
                p = pid_np[vi][mask_np[vi]]
                qv = q_np[vi][mask_np[vi]]
                tot = float(qv.sum())
                if tot > 0.0:
                    bc = np.bincount(p, weights=qv, minlength=n_groups)
                    frac[vi, :] = (bc / tot).astype(np.float32)
                    voxel_pid[vi] = int(np.argmax(bc))

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
            extra["voxel_best_tube"] = tid[torch.arange(M, device=voxels.device), bestj].to(torch.int32).cpu().numpy()

        if have_pe and idx_pe is not None:
            pe_vals = voxels[..., idx_pe]
            pe_sums = (pe_vals * mask.to(pe_vals.dtype)).sum(dim=1)
            extra["voxel_n_pe_sum"] = pe_sums.cpu().numpy().astype(np.float32, copy=False)

        return feats_out, extra

    def _apply_norm(self, feats: np.ndarray):
        if self._mean is None:
            return feats
        cols = self.feat_norm_idx
        x = feats[:, cols]
        if self.feat_norm == "standard":
            x = (x - self._mean) / np.clip(self._std, 1e-6, None)
        elif self.feat_norm == "minmax":
            x = (x - self._fmin) / np.clip(self._fmax - self._fmin, 1e-6, None)
        feats[:, cols] = x
        return feats

    def _accum_stats(self, feats: np.ndarray, s: Dict[str, np.ndarray]):
        if feats.shape[0] == 0:
            return
        x = feats[:, self.feat_norm_idx].astype(np.float64)
        s["n"] += x.shape[0]
        s["sum"] += x.sum(0)
        s["sum2"] += (x * x).sum(0)
        s["min"] = np.minimum(s["min"], x.min(0))
        s["max"] = np.maximum(s["max"], x.max(0))

    def _stats_signature(self) -> str:
        g = self.grid
        return "|".join([
            f"num_batches={self.num_batches}",
            f"seed={self.seed}",
            f"stats_max_events={self.stats_max_events}",
            f"axis={g.axis_limit}",
            f"grid={g.grid_size}",
            f"minq={self.min_charge}",
        ])

    def _try_load_feature_stats(self) -> bool:
        p = self._stats_cache_path
        if p is None or (not os.path.isfile(p)):
            return False
        z = np.load(p, allow_pickle=False)
        if str(z["signature"]) != self._stats_signature():
            return False
        self._mean = z["mean"].astype(np.float32)
        self._std = z["std"].astype(np.float32)
        self._fmin = z["fmin"].astype(np.float32)
        self._fmax = z["fmax"].astype(np.float32)
        log.info(f"Loaded feature stats cache: {p}")
        return True

    def _try_save_feature_stats(self) -> bool:
        p = self._stats_cache_path
        if p is None:
            return False
        try:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            np.savez_compressed(p, signature=self._stats_signature(), mean=self._mean, std=self._std, fmin=self._fmin, fmax=self._fmax)
        except OSError as e:
            log.warning(
                f"Could not write feature stats cache to {p} ({e}); continuing with the "
                f"in-memory stats. Set data.dataset.params.stats_cache_path to a writable "
                f"location to persist and reuse it across runs."
            )
            return False
        log.info(f"Saved feature stats cache: {p}")
        return True

    def _init_feature_stats(self, stats_max_events: Optional[int]):
        N = len(self)
        take = N if (stats_max_events is None) else min(N, int(stats_max_events))
        idxs = np.arange(N)
        self.rng.shuffle(idxs)
        idxs = idxs[:take]

        by_file: Dict[int, List[int]] = {}
        for j in idxs:
            fidx, eidx = self._index[int(j)]
            by_file.setdefault(int(fidx), []).append(int(eidx))

        k = len(self.feat_norm_idx)
        s = {
            "n": 0.0,
            "sum": np.zeros((k,), dtype=np.float64),
            "sum2": np.zeros((k,), dtype=np.float64),
            "min": np.full((k,), np.inf, dtype=np.float64),
            "max": np.full((k,), -np.inf, dtype=np.float64),
        }

        for fidx, eidxs in by_file.items():
            with h5py.File(self.files[int(fidx)], "r") as f:
                for eidx in eidxs:
                    digi_hits = self._read_digi_event(f, int(eidx))
                    if digi_hits.shape[0] == 0:
                        continue

                    xyz = np.stack([digi_hits["x"], digi_hits["y"], digi_hits["z"]], axis=1).astype(np.float32)
                    q = digi_hits["charge_pe"].astype(np.float32)
                    t = digi_hits["time_ns"].astype(np.float32)

                    feats3, extra3 = self._voxelize_spconv_3d(xyz, q, t)
                    mv = feats3[:, 0] >= self.min_charge
                    feats3 = feats3[mv]
                    coords_zyx = extra3["coords_zyx"][mv]
                    if feats3.shape[0] == 0:
                        continue

                    centers = self._voxel_centers(coords_zyx)
                    feats = np.concatenate([feats3[:, 0:1], centers, feats3[:, 1:2]], axis=1).astype(np.float32)
                    self._accum_stats(feats, s)

        n = float(max(1.0, s["n"]))
        mean = s["sum"] / n
        var = s["sum2"] / n - mean * mean
        self._mean = mean.astype(np.float32)
        self._std = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)
        self._fmin = s["min"].astype(np.float32)
        self._fmax = s["max"].astype(np.float32)

    def __del__(self):
        self.close()


class HyperKSparseCNN3D(_BaseSparseDataset):
    def __getitem__(self, i: int) -> Dict[str, Any]:
        if self.cache_in_ram and i in self._cache:
            return self._cache[i]

        fidx, eidx = self._index[int(i)]
        f = self._get_h5(fidx)

        digi_hits_raw = self._read_digi_event(f, eidx)
        photon_ids_flat_raw, photon_ptr_raw = self._read_digi_photon_lists_event(f, eidx)

        if digi_hits_raw.shape[0] > 0:
            key = digi_hits_raw["tube_id"].astype(np.int64, copy=False)
            time = digi_hits_raw["time_ns"].astype(np.float64, copy=False)

            order = np.lexsort((time, key))
            first = np.r_[True, key[order][1:] != key[order][:-1]]
            keep_idx = np.sort(order[first]).astype(np.int64, copy=False)

            digi_hits = digi_hits_raw[keep_idx]

            chunks = [
                photon_ids_flat_raw[int(photon_ptr_raw[j]):int(photon_ptr_raw[j + 1])]
                for j in keep_idx
            ]
            photon_ids_flat = np.concatenate(chunks).astype(np.int64, copy=False) if chunks else np.zeros((0,), dtype=np.int64)
            photon_ptr = np.r_[0, np.cumsum([len(c) for c in chunks])].astype(np.int64)
        else:
            digi_hits = digi_hits_raw
            photon_ids_flat = np.zeros((0,), dtype=np.int64)
            photon_ptr = np.zeros((1,), dtype=np.int64)

        xyz = np.stack([digi_hits["x"], digi_hits["y"], digi_hits["z"]], axis=1).astype(np.float32)
        q = digi_hits["charge_pe"].astype(np.float32)
        t = digi_hits["time_ns"].astype(np.float32)
        tubes = digi_hits["tube_id"].astype(np.int32, copy=False) if _has_field(digi_hits, "tube_id") else np.full((digi_hits.shape[0],), -1, dtype=np.int32)

        rot_angle = float(self.rng.uniform(0.0, 2.0 * math.pi)) if self.augmentation else 0.0
        if self.augmentation and xyz.shape[0] > 0:
            _rotate_xy_inplace(xyz, rot_angle)

        hits, parts = self._read_event(f, eidx)

        q_label = hits["charge_pe"].astype(np.float32) if _has_field(hits, "charge_pe") else np.ones((hits.shape[0],), dtype=np.float32)

        digi_n_photons = np.zeros((digi_hits.shape[0],), dtype=np.int32)
        if photon_ptr.size == digi_hits.shape[0] + 1:
            digi_n_photons = np.diff(photon_ptr).astype(np.int32, copy=False)

        if self.keep_in_digit and hits.shape[0] > 0 and photon_ids_flat.size > 0:
            mask_used = np.zeros((hits.shape[0],), dtype=bool)
            good = (photon_ids_flat >= 0) & (photon_ids_flat < hits.shape[0])
            mask_used[photon_ids_flat[good]] = True
            q_label = q_label * mask_used.astype(np.float32)

        pid_hits, top_tids, top_pdgs, top_info = self._assign_primary_hits(
            hits, parts, q_label, topk=self.max_parents, return_info=self.add_meta
        )
        pid_digi = self._assign_primary_digis(q, photon_ids_flat, photon_ptr, pid_hits)

        tube_ids = tubes if self.add_meta else None
        feats3, extra3 = self._voxelize_spconv_3d(xyz, q, t, parent_ids=pid_digi, tube_ids=tube_ids)

        coords_zyx = extra3["coords_zyx"]
        voxel_parent_np = extra3["voxel_parent_id"]
        voxel_parent_frac = extra3["voxel_parent_frac"]

        mv = feats3[:, 0] >= self.min_charge
        feats3 = feats3[mv]
        coords_zyx = coords_zyx[mv]
        voxel_parent_np = voxel_parent_np[mv]
        voxel_parent_frac = voxel_parent_frac[mv]

        voxel_best_tube = extra3["voxel_best_tube"][mv] if (self.add_meta and "voxel_best_tube" in extra3) else None

        centers = self._voxel_centers(coords_zyx)
        feats_np = np.concatenate([feats3[:, 0:1], centers, feats3[:, 1:2]], axis=1).astype(np.float32)

        raw_charge_pe_np = feats_np[:, 0].copy() if self.add_meta else None
        raw_time_ns_np = feats_np[:, -1].copy() if self.add_meta else None

        feats = self._apply_norm(feats_np)
        coords = np.concatenate([np.zeros((coords_zyx.shape[0], 1), dtype=np.int32), coords_zyx], axis=1).astype(np.int32, copy=False)

        voxel_parent = voxel_parent_np.astype(np.int64, copy=False)
        voxel_parent_frac = voxel_parent_frac.astype(np.float32, copy=False)

        G_full = self.max_parents + 1
        if voxel_parent_frac.shape[1] < G_full:
            pad = np.zeros((voxel_parent_frac.shape[0], G_full - voxel_parent_frac.shape[1]), dtype=np.float32)
            voxel_parent_frac = np.concatenate([voxel_parent_frac, pad], axis=1)
        elif voxel_parent_frac.shape[1] > G_full:
            voxel_parent_frac = voxel_parent_frac[:, :G_full]

        meta = {
            "voxel_parent_id": voxel_parent,
            "voxel_parent_frac": voxel_parent_frac,
            "grid_size": (self.grid.grid_size, self.grid.grid_size, self.grid.grid_size),
            "axis_limit": self.grid.axis_limit,
            "voxel_size": self.grid.voxel_size,
            "origin": self.grid.origin.copy(),
        }

        if self.add_meta:
            vertex = np.array([parts["vtx_x"][0], parts["vtx_y"][0], parts["vtx_z"][0]], dtype=np.float32) if len(parts) > 0 else np.zeros((3,), dtype=np.float32)
            if self.augmentation and top_info:
                for d in top_info:
                    dx, dy, dz = d.get("dir", (0.0, 0.0, 0.0))
                    vv2 = _rotate_xy_vec3(np.array([dx, dy, dz], dtype=np.float32), rot_angle)
                    n = float(np.linalg.norm(vv2))
                    if n > 0:
                        vv2 = vv2 / n
                    d["dir"] = (float(vv2[0]), float(vv2[1]), float(vv2[2]))
            if self.augmentation:
                v2 = _rotate_xy_vec3(np.array([vertex[0], vertex[1], 0.0], dtype=np.float32), rot_angle)
                vertex[0], vertex[1] = v2[0], v2[1]

            pmt_n_photons = {}
            for tube_id, nph in zip(tubes.tolist(), digi_n_photons.tolist()):
                tube_id = int(tube_id)
                if tube_id < 0:
                    continue
                pmt_n_photons[tube_id] = pmt_n_photons.get(tube_id, 0) + int(nph)

            meta.update({
                "event_index": eidx,
                "file_path": self.files[fidx],
                "events_in_file": int(self._events_per_file[self.files[fidx]]),
                "events_are_separate": True,
                
                "vertex": vertex,
                "voxel_charge_pe": raw_charge_pe_np.astype(np.float32, copy=False),
                "voxel_time_ns": raw_time_ns_np.astype(np.float32, copy=False),
                "pmt_n_photons": pmt_n_photons,
                "digi_n_photons": digi_n_photons.astype(np.int32, copy=False),
                "digi_tube_ids": tubes.astype(np.int32, copy=False),
                "parent_pdgs": top_pdgs,
                "parent_track_ids": top_tids,
                "parent_info": top_info,
                "max_parents": self.max_parents,
                "true_time_ns": hits["time_ns"].astype(np.float32) if _has_field(hits, "time_ns") else np.zeros((hits.shape[0],), dtype=np.float32),
            })

            meta.update(self._read_trigger_event(f, eidx))

            pmt_to_voxel = {}
            if voxel_best_tube is not None:
                pmt_to_voxel = {int(t): int(vi) for vi, t in enumerate(voxel_best_tube.tolist()) if int(t) >= 0}
                meta["voxel_best_tube"] = voxel_best_tube.astype(np.int32, copy=False)
            meta["pmt_to_voxel"] = pmt_to_voxel

        sample = {"coords": coords, "feats": feats, "meta": meta}
        if self.cache_in_ram:
            self._cache[i] = sample
        return sample
