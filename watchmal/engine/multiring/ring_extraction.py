# -*- coding: utf-8 -*-
"""
Segmented rings from the multi-ring network.

Three charge-split methods, selected by RingConfig.split_method:
  map    probability head: dominant voxels keep their charge, mixed voxels split
         by fraction and re-digitize from the MAP photon count N = argmax P(N|q).
  bayes  probability head: like map, but each mixed voxel infers N from the
         empirical posterior P(N | q*frac) ~ P(q*frac | N) P(N), then draws its
         charge from P(Q | N) -- the P(N) prior corrects MAP's 1->2 PE migration.
  count  PE-count head: the network already gives each ring its photon rate, so
         only the final digitization is left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class Ring:
    """One reconstructed ring: the PMTs it owns and the charge it carries."""
    ring_id: int
    tube_id: np.ndarray     # (H,) int64
    charge_pe: np.ndarray   # (H,) float32
    time_ns: np.ndarray     # (H,) float32

    @property
    def total_charge(self) -> float:
        return float(self.charge_pe.sum())


@dataclass
class RingConfig:
    """Everything that changes the rings, in one place."""
    split_method: str = "map"            # map | bayes | count
    ring_pe_threshold: float = 80.0      # a ring exists iff it carries this much charge
    dominant_frac_threshold: float = 0.99
    min_charge_pe: float = 1e-4
    max_rings: int = 4
    background_charge_scale: float = 1.0
    noise_fraction: float = 1.0
    fold_background: bool = True
    bayes_pred_threshold: float = 0.24   # PE — mixed entries below this are dropped (bayes)
    bayes_n_max: int = 20
    bayes_q_max: float = 25.0
    bayes_n_bins: int = 5000
    seed: int = 0

    @classmethod
    def from_mapping(cls, cfg: Dict[str, Any]) -> "RingConfig":
        f = cls()
        for k in cls.__dataclass_fields__:
            if k in cfg and cfg[k] not in (None, ""):
                setattr(f, k, type(getattr(f, k))(cfg[k]))
        return f


def fold_background(raw_frac: np.ndarray, bg_frac: np.ndarray, scale: float,
                    noise_fraction: float, rng) -> np.ndarray:
    """
    Give every active ring the full background fraction, not a share of it.

    Each segmented ring should look like a single-ring event, which would see
    all of the event's dark noise on its own. So the background is duplicated
    across rings rather than divided between them.
    """
    if scale == 0.0 or noise_fraction <= 0.0:
        return raw_frac.astype(np.float32, copy=True)
    out = raw_frac.astype(np.float32, copy=True)
    active = raw_frac.sum(axis=0) > 0
    if not active.any():
        return out
    if noise_fraction < 1.0:
        keep = rng.random(raw_frac.shape[0]) < noise_fraction
        bg_frac = bg_frac * keep.astype(np.float32)
    out[:, active] += (bg_frac[:, None] * scale)
    return out


class BayesianResampler:
    """
    q*frac -> N ~ P(N | q*frac) ~ P(q*frac | N) P(N) -> Q ~ P(Q | N).

    Built from the digitizer's empirical mapping {N: q_samples}: the likelihood
    P(q*frac | N) is the normalised histogram of q_samples evaluated at q*frac,
    the prior P(N) is the sample frequency of each N. The photon count N is drawn
    from the posterior, then the charge is redrawn from P(Q | N=N). Folding in the
    strong P(N=1) prior corrects MAP's 1->2 PE migration.
    """

    def __init__(self, mapping: Dict[int, np.ndarray], n_max: int = 20,
                 q_max: float = 25.0, n_bins: int = 5000, seed: int = 0):
        self.n_bins = int(n_bins)
        self.edges = np.linspace(0.0, float(q_max), self.n_bins + 1)
        self._rng = np.random.default_rng(int(seed))
        self.q_samples = {int(n): np.asarray(mapping[n], np.float32)
                          for n in mapping if 1 <= int(n) <= int(n_max)}

        counts = np.zeros(int(n_max) + 1)
        pqn = np.zeros((int(n_max) + 1, self.n_bins))
        for n, q_arr in self.q_samples.items():
            counts[n] = len(q_arr)
            h, _ = np.histogram(q_arr, bins=self.edges)
            if h.sum() > 0:
                pqn[n] = h / h.sum()
        if counts.sum() <= 0:
            raise RuntimeError("BayesianResampler: empty digitizer mapping")

        prior = counts / counts.sum()
        unnorm = pqn * prior[:, None]
        posterior = unnorm / np.maximum(unnorm.sum(axis=0)[None, :], 1e-12)
        self.cum_post = np.cumsum(posterior, axis=0)

    def resample(self, lam: np.ndarray) -> np.ndarray:
        """lam -> N (Bayesian) -> Q (re-digitize)."""
        if len(lam) == 0:
            return np.zeros(0, np.float32)
        idx = np.clip(np.searchsorted(self.edges, lam) - 1, 0, self.n_bins - 1)
        u = self._rng.random(len(lam))
        N = (self.cum_post[:, idx].T >= u[:, None]).argmax(axis=1).astype(np.int32)

        q_out = np.zeros(len(N), np.float32)
        for n in np.unique(N):
            n_int = int(n)
            if n_int == 0:
                continue
            mask = N == n_int
            samp = self.q_samples.get(n_int)
            if samp is not None:
                q_out[mask] = samp[self._rng.integers(0, len(samp), int(mask.sum()))]
            else:
                mean_q = 9.652 + (n_int - 9) * 1.003
                sigma = 0.5 * np.sqrt(max(n_int, 1))
                q_out[mask] = (mean_q + self._rng.normal(0, sigma, int(mask.sum()))).astype(np.float32)
        return q_out


class RingSegmentor:
    """
    Network output in, rings out.

    seg = RingSegmentor(model, voxelizer, digitizer, RingConfig(...))
    rings = seg.rings(xyz, charge_pe, time_ns, tube_ids)
    """

    def __init__(self, model, voxelizer, digitizer, config: Optional[RingConfig] = None,
                 device: Optional[str] = None):
        self.model = model
        self.voxelizer = voxelizer
        self.digitizer = digitizer
        self.cfg = config or RingConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._np_rng = np.random.RandomState(self.cfg.seed)

        self.bayes: Optional[BayesianResampler] = None
        if self.cfg.split_method == "bayes":
            self.bayes = BayesianResampler(
                self.digitizer.mapping, n_max=self.cfg.bayes_n_max,
                q_max=self.cfg.bayes_q_max, n_bins=self.cfg.bayes_n_bins,
                seed=self.cfg.seed)

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def _forward(self, coords: np.ndarray, feats: np.ndarray, grid) -> Dict[str, Any]:
        c = coords.copy()
        c[:, 0] = 0
        batch = {"coords": torch.from_numpy(c).to(self.device),
                 "feats": torch.from_numpy(feats).to(self.device),
                 "meta": [{"grid_size": (grid.grid_size,) * 3,
                           "axis_limit": grid.axis_limit,
                           "voxel_size": grid.voxel_size}]}
        return self.model(batch)

    @staticmethod
    def _normalise(t: torch.Tensor) -> np.ndarray:
        t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return t.detach().cpu().numpy().astype(np.float32)

    # -------------------------------------------------------------- rings
    def rings(self, xyz, charge_pe, time_ns, tube_ids, min_charge: float = 0.01) -> List[Ring]:
        """Voxelize the hits, run the network, return the rings it finds."""
        coords, feats, extra = self.voxelizer.build_model_input(
            np.asarray(xyz, np.float32), np.asarray(charge_pe, np.float32),
            np.asarray(time_ns, np.float32), np.asarray(tube_ids, np.int64),
            min_charge=min_charge)
        if coords.shape[0] == 0:
            return []

        out = self._forward(coords, feats, self.voxelizer.grid)
        q = extra["voxel_q"]
        t = extra["voxel_t"]
        tubes = extra["voxel_best_tube"].astype(np.int64)

        if self.cfg.split_method == "count":
            if not out.get("LogLambda_list"):
                raise ValueError("split_method='count' needs a PE-count head (LogLambda_list)")
            return self.rings_from_count(out["LogLambda_list"][0], q, t, tubes)

        if not out.get("Pi_list"):
            raise ValueError(f"split_method='{self.cfg.split_method}' needs a probability head (Pi_list)")
        if self.cfg.split_method == "map":
            return self.rings_from_map(out["Pi_list"][0], q, t, tubes)
        if self.cfg.split_method == "bayes":
            return self.rings_from_bayes(out["Pi_list"][0], q, t, tubes)
        raise ValueError(f"unknown split_method: {self.cfg.split_method!r}")

    def _fracs(self, Pi: torch.Tensor):
        """Normalised ring fractions, background, and the bg-folded split weights."""
        P = self._normalise(Pi)
        raw = P[:, 1:]
        bg = P[:, 0]
        s = np.clip(raw.sum(axis=1), 1e-6, None)
        frac = (raw / s[:, None]).astype(np.float32)
        split = (fold_background(raw, bg, self.cfg.background_charge_scale,
                                 self.cfg.noise_fraction, self.rng)
                 if self.cfg.fold_background else raw)
        return raw, bg, frac, split

    def _dominant(self, frac: np.ndarray, present: List[int]):
        mat = np.stack([frac[:, k] for k in present], axis=1)
        dominated = mat.max(axis=1) >= self.cfg.dominant_frac_threshold
        return dominated, mat.argmax(axis=1)

    def rings_from_map(self, Pi: torch.Tensor, q, t, tubes) -> List[Ring]:
        """map: dominant voxels keep q, mixed voxels re-digitized from MAP(q)*frac."""
        raw, bg, frac, split = self._fracs(Pi)
        present = self._present_rings(raw, bg, q)
        if not present:
            return []

        n_photons = self.digitizer.map_n_from_q_array(q).astype(np.float32)
        dominated, owner = self._dominant(frac, present)
        shared = ~dominated

        rings = []
        for i, k in enumerate(present):
            qk = np.zeros(q.shape[0], np.float32)
            qk[dominated & (owner == i)] = q[dominated & (owner == i)]
            if shared.any():
                lam = (n_photons * split[:, k])[shared]
                qk[shared] = self.digitizer.stochastic_round_then_digitize(
                    lam.astype(np.float32)).astype(np.float32)
            r = self._make_ring(k, qk, q, t, tubes)
            if r is not None:
                rings.append(r)
        return rings

    def rings_from_bayes(self, Pi: torch.Tensor, q, t, tubes) -> List[Ring]:
        """bayes: dominant voxels keep q, mixed voxels resampled from P(N | q*frac)."""
        raw, bg, frac, split = self._fracs(Pi)
        present = self._present_rings(raw, bg, q)
        if not present:
            return []

        dominated, owner = self._dominant(frac, present)
        shared = ~dominated

        rings = []
        for i, k in enumerate(present):
            qk = np.zeros(q.shape[0], np.float32)
            qk[dominated & (owner == i)] = q[dominated & (owner == i)]
            if shared.any():
                lam = (q * split[:, k])[shared]
                keep = lam >= self.cfg.bayes_pred_threshold
                if keep.any():
                    idx = np.where(shared)[0][keep]
                    qk[idx] = self.bayes.resample(lam[keep]).astype(np.float32)
            r = self._make_ring(k, qk, q, t, tubes)
            if r is not None:
                rings.append(r)
        return rings

    def rings_from_count(self, log_lambda: torch.Tensor, q, t, tubes) -> List[Ring]:
        """count: the network gives each ring its photon rate, only digitization is left."""
        lam = torch.exp(log_lambda.clamp(max=20.0))
        lam = torch.nan_to_num(lam, nan=0.0, posinf=0.0, neginf=0.0)
        L = lam.detach().cpu().numpy().astype(np.float32)[:, 1:]

        present = [k for k in range(min(self.cfg.max_rings, L.shape[1]))
                   if float(L[:, k].sum()) >= self.cfg.ring_pe_threshold]
        rings = []
        for k in present:
            n = self._np_rng.poisson(np.clip(L[:, k], 0.0, None)).astype(np.int32)
            qk = np.array([self.digitizer.sample_q_given_n(int(v)) for v in n], np.float32)
            r = self._make_ring(k, qk, q, t, tubes)
            if r is not None:
                rings.append(r)
        return rings

    # ------------------------------------------------------------- helpers
    def _present_rings(self, raw: np.ndarray, bg: np.ndarray, q: np.ndarray) -> List[int]:
        owner = np.concatenate([bg[:, None], raw], axis=1).argmax(axis=1)
        return [k for k in range(min(self.cfg.max_rings, raw.shape[1]))
                if float(q[owner == k + 1].sum()) >= self.cfg.ring_pe_threshold]

    def _make_ring(self, k: int, qk, q, t, tubes) -> Optional[Ring]:
        keep = (tubes > 0) & (qk >= self.cfg.min_charge_pe)
        if not keep.any():
            return None
        return Ring(ring_id=k + 1, tube_id=tubes[keep],
                    charge_pe=qk[keep].astype(np.float32),
                    time_ns=t[keep].astype(np.float32))
