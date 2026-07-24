# -*- coding: utf-8 -*-
"""
Empirical WCSim digitizer.

lambda_true -> N_true = round(lambda) -> Q_digi ~ P(Q | N_true), and the inverse
MAP(N | q_digi). Needed by RingSegmentor to re-digitize the per-ring charge
split; P(Q | N) comes from the .npz produced by truehit_to_digihit_mapping.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def stochastic_round_charge(q: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Unbiased stochastic rounding: q = n + f -> n+1 w.p. f, else n."""
    q = np.asarray(q, dtype=np.float32)
    q = np.where(np.isfinite(q), q, 0.0).astype(np.float32, copy=False)
    q = np.clip(q, 0.0, None).astype(np.float32, copy=False)
    q_floor = np.floor(q).astype(np.float32, copy=False)
    add_one = rng.rand(*q.shape) < (q - q_floor)
    return (q_floor + add_one.astype(np.float32)).astype(np.int32, copy=False)


def nearest_int_round_charge(q: np.ndarray) -> np.ndarray:
    """Nearest-integer (banker's) rounding."""
    q = np.asarray(q, dtype=np.float32)
    q = np.where(np.isfinite(q), q, 0.0).astype(np.float32, copy=False)
    q = np.clip(q, 0.0, None).astype(np.float32, copy=False)
    return np.round(q).astype(np.int32, copy=False)


class WCSimDigitizer:
    """Empirical true-hit to digihit digitizer built from an .npz mapping."""

    def __init__(
        self,
        mapping_path: str,
        seed: int = 0,
        *,
        rescale_unseen_n: bool = True,
        use_n0_mapping: bool = False,
        remove_zero_q_bins: bool = True,
        max_direct_n: int = 10_000,
        clip_negative: bool = True,
        use_stochastic_rounding: bool = True,
        use_optimized_lut: bool = False,
        lut_crossovers_path: str = "",
        lut_q_base: float = 9.652,
        lut_n_base: int = 9,
        lut_spacing: float = 1.003,
    ) -> None:
        self.mapping_path = str(mapping_path)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.rescale_unseen_n = bool(rescale_unseen_n)
        self.use_n0_mapping = bool(use_n0_mapping)
        self.remove_zero_q_bins = bool(remove_zero_q_bins)
        self.max_direct_n = int(max_direct_n)
        self.clip_negative = bool(clip_negative)
        self.use_stochastic_rounding = bool(use_stochastic_rounding)
        self.use_optimized_lut = bool(use_optimized_lut)
        self.lut_crossovers_path = str(lut_crossovers_path)
        self.lut_q_base = float(lut_q_base)
        self.lut_n_base = int(lut_n_base)
        self.lut_spacing = float(lut_spacing)

        self.mapping: Dict[int, np.ndarray] = {}
        self.mode = ""

        if not self.mapping_path:
            raise ValueError("Digitizer mapping_path is empty.")
        path = Path(self.mapping_path)
        if not path.exists():
            raise FileNotFoundError(f"Digitizer mapping file not found: {path}")
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Digitizer mapping must be a .npz file, got: {path}")
        self._load_npz(path)

        self.available_n = np.asarray(sorted(self.mapping.keys()), dtype=np.int32)
        if self.available_n.size == 0:
            raise RuntimeError(f"Digitizer mapping is empty: {path}")

        q_n1 = self.mapping.get(1)
        self.mean_charge_per_photon: float = (
            float(np.mean(q_n1)) if q_n1 is not None and len(q_n1) > 0 else 1.0
        )
        self._build_map_lut()

    def _load_npz(self, path: Path) -> None:
        data = np.load(str(path), allow_pickle=False)
        if "n_values" not in data:
            raise RuntimeError(f"NPZ digitizer mapping has no 'n_values': {path}")

        mapping: Dict[int, np.ndarray] = {}
        for n in data["n_values"].astype(np.int32).reshape(-1).tolist():
            key = f"q_values_{int(n)}"
            if key not in data:
                continue
            q = np.asarray(data[key], dtype=np.float32).reshape(-1)
            q = q[np.isfinite(q)]
            if self.clip_negative:
                q = np.clip(q, 0.0, None)
            if self.remove_zero_q_bins:
                q = q[q > 0.0]
            if q.size > 0:
                mapping[int(n)] = q.astype(np.float32, copy=False)

        self.mapping = mapping
        self.mode = "npz_empirical"

    def _nearest_available_n(self, n: int) -> int:
        idx = int(np.argmin(np.abs(self.available_n.astype(np.int64) - int(n))))
        return int(self.available_n[idx])

    def _sample_from_empirical(self, n: int) -> float:
        n = int(n)
        if n <= 0 and not self.use_n0_mapping:
            return 0.0

        scale = 1.0
        key = n
        if key not in self.mapping:
            key = self._nearest_available_n(n)
            if self.rescale_unseen_n and key > 0:
                scale = float(n) / float(key)

        q = float(self.mapping[key][self.rng.randint(0, int(self.mapping[key].size))]) * scale
        if self.clip_negative and q < 0.0:
            q = 0.0
        return float(q)

    def _build_map_lut(self, q_max: float = 25.0, n_bins: int = 5000) -> None:
        """MAP N-estimate per q_digi bin (N >= 1): raw histogram argmax, or crossover LUT."""
        q_edges = np.linspace(0.0, q_max, n_bins + 1, dtype=np.float64)
        centers = 0.5 * (q_edges[:-1] + q_edges[1:])
        n_list = sorted(k for k in self.mapping if k >= 1)

        if self.use_optimized_lut:
            lut = self._build_crossover_lut_array(centers, n_bins, q_max)
        else:
            prob_matrix = np.zeros((len(n_list), n_bins), dtype=np.float64)
            for i, n in enumerate(n_list):
                counts, _ = np.histogram(self.mapping[n].astype(np.float64), bins=q_edges)
                total = float(counts.sum())
                if total > 0:
                    prob_matrix[i] = counts / total
            lut = np.array(n_list, dtype=np.int32)[np.argmax(prob_matrix, axis=0)]

        self._map_lut = lut
        self._map_q_edges = q_edges

    def _build_crossover_lut_array(self, centers: np.ndarray, n_bins: int, q_max: float) -> np.ndarray:
        """Step-function MAP LUT from crossovers in lut_crossovers_path, linearly extended past the last."""
        crossovers: list = []
        if self.lut_crossovers_path:
            path = Path(self.lut_crossovers_path)
            if not path.exists():
                raise FileNotFoundError(f"lut_crossovers_path not found: {path}")
            with path.open() as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    parts = s.split()
                    if len(parts) < 2:
                        continue
                    crossovers.append((float(parts[1]), int(parts[0])))
            crossovers.sort(key=lambda x: x[0])

        last_n = crossovers[-1][1] if crossovers else 1
        k = 1
        while True:
            N = last_n + k - 1
            cross_q = self.lut_q_base + (N - self.lut_n_base) * self.lut_spacing
            if cross_q >= q_max:
                break
            crossovers.append((cross_q, N + 1))
            k += 1

        lut = np.ones(n_bins, dtype=np.int32)
        for cross_q, n_above in crossovers:
            lut[centers >= cross_q] = n_above
        return lut

    def map_n_from_q_array(self, q_digi: np.ndarray) -> np.ndarray:
        """Vectorized MAP estimate N_true = argmax_{N>=1} P(q_digi | N)."""
        q = np.asarray(q_digi, dtype=np.float64).reshape(-1)
        idx = np.clip(np.searchsorted(self._map_q_edges, q, side="right") - 1, 0, len(self._map_lut) - 1)
        result = self._map_lut[idx].copy()
        result[q <= 0.0] = 0

        beyond = q > self._map_q_edges[-1]
        if beyond.any():
            mean = self.mean_charge_per_photon if self.mean_charge_per_photon > 0 else 1.0
            result[beyond] = np.maximum(1, np.round(q[beyond] / mean).astype(np.int32))
        return result.astype(np.int32)

    def sample_q_given_n(self, n_true_pe: int) -> float:
        """Sample Q_digi from P(Q_digi | N_true_PE)."""
        n = int(n_true_pe)
        if n <= 0 and not self.use_n0_mapping:
            return 0.0
        return self._sample_from_empirical(n)

    def stochastic_round_then_digitize(self, lambda_true_pe: np.ndarray) -> np.ndarray:
        """N_true = round(lambda) [stochastic/nearest], then Q_digi ~ P(Q | N_true), grouped by N."""
        lam = np.asarray(lambda_true_pe, dtype=np.float32)
        lam = np.where(np.isfinite(lam), lam, 0.0).astype(np.float32, copy=False)
        lam = np.clip(lam, 0.0, None).astype(np.float32, copy=False)

        if self.use_stochastic_rounding:
            n_true = stochastic_round_charge(lam, self.rng)
        else:
            n_true = nearest_int_round_charge(lam)

        flat_n = np.clip(n_true.reshape(-1).astype(np.int32), 0, self.max_direct_n)
        flat_out = np.zeros(flat_n.size, dtype=np.float32)

        for n_val in np.unique(flat_n):
            n = int(n_val)
            mask = flat_n == n
            count = int(mask.sum())
            if n <= 0 and not self.use_n0_mapping:
                continue

            scale = 1.0
            key = n
            if key not in self.mapping:
                key = self._nearest_available_n(n)
                if self.rescale_unseen_n and key > 0 and n > 0:
                    scale = float(n) / float(key)

            qvals = self.mapping[key]
            samples = qvals[self.rng.randint(0, int(qvals.size), size=count)].astype(np.float32)
            if scale != 1.0:
                samples = samples * np.float32(scale)
            if self.clip_negative:
                samples = np.maximum(samples, np.float32(0.0))
            flat_out[mask] = samples

        return flat_out.reshape(lam.shape).astype(np.float32, copy=False)

    def __call__(self, lambda_true_pe: np.ndarray) -> np.ndarray:
        return self.stochastic_round_then_digitize(lambda_true_pe)


def build_digitizer(mapping_path: str, seed: int = 0, **kwargs: Any) -> Optional[WCSimDigitizer]:
    """Build a WCSimDigitizer, or None if mapping_path is empty."""
    if mapping_path is None or str(mapping_path).strip() == "":
        return None
    return WCSimDigitizer(mapping_path=str(mapping_path), seed=int(seed), **kwargs)


def digitize_charge(
    q_pred: np.ndarray,
    digitizer: Optional[WCSimDigitizer],
    rng: Optional[np.random.RandomState] = None,
    use_stochastic_rounding_without_digitizer: bool = False,
) -> np.ndarray:
    q = np.asarray(q_pred, dtype=np.float32)
    q = np.where(np.isfinite(q), q, 0.0).astype(np.float32, copy=False)
    q = np.clip(q, 0.0, None).astype(np.float32, copy=False)

    if digitizer is not None:
        return digitizer.stochastic_round_then_digitize(q)
    if use_stochastic_rounding_without_digitizer:
        if rng is None:
            rng = np.random.RandomState(0)
        return stochastic_round_charge(q, rng).astype(np.float32, copy=False)
    return q.astype(np.float32, copy=False)
