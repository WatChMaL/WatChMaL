# -*- coding: utf-8 -*-
"""
Batch sampler for MultiRing sparse-CNN datasets.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import numpy as np
from torch.utils.data import Subset, Sampler

class BatchFileSampler(Sampler[List[int]]):
    def __init__(
        self,
        subset: Subset,
        batch_size: int,
        *,
        shuffle_files: bool = False,
        shuffle_events_within_file: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(None)
        if not isinstance(subset, Subset):
            raise TypeError("BatchFileSampler expects a torch.utils.data.Subset")
        self.subset = subset
        self.full_ds = subset.dataset
        self.subset_indices: List[int] = list(map(int, subset.indices))
        self.batch_size = int(batch_size)
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_events_within_file = bool(shuffle_events_within_file)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.epoch = 0
        self._order: Optional[List[int]] = None

        self.N = len(self.subset_indices)
        if self.N <= 0:
            raise ValueError("Empty subset.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.steps_per_epoch = self.N // self.batch_size
        if (not self.drop_last) and (self.N % self.batch_size != 0):
            self.steps_per_epoch += 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._order = None

    def _build_order(self) -> List[int]:
        by_file: Dict[int, List[int]] = {}
        for sub_i, global_i in enumerate(self.subset_indices):
            fidx, _eidx = self.full_ds._index[int(global_i)]
            by_file.setdefault(int(fidx), []).append(int(sub_i))

        files = sorted(by_file.keys())
        rng = np.random.RandomState(self.seed + self.epoch)

        if self.shuffle_files:
            rng.shuffle(files)

        order: List[int] = []
        for fidx in files:
            idxs = by_file[fidx]
            if self.shuffle_events_within_file:
                idxs = idxs.copy()
                rng.shuffle(idxs)
            order.extend(idxs)

        return order

    def __iter__(self) -> Iterator[List[int]]:
        if self._order is None:
            self._order = self._build_order()

        order = self._order
        N = self.N
        B = self.batch_size

        start = (self.epoch * self.steps_per_epoch * B) % N

        for s in range(self.steps_per_epoch):
            pos0 = (start + s * B) % N
            batch = [order[(pos0 + k) % N] for k in range(B)]
            if self.drop_last and len(batch) < B:
                continue
            yield batch

    def __len__(self) -> int:
        return self.steps_per_epoch
