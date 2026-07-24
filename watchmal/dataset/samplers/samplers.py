"""
Sampler classes
"""

# torch imports
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

# generic imports
from operator import itemgetter
from typing import Optional

from typing import Iterator, Optional, Sized, Sequence


class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        indices (sequence): a sequence to sample (sequentially) from
        generator: accepted so every sampler shares one interface, and ignored:
            the order is fixed, so there is nothing to draw.
        device: idem.
    """

    def __init__(self, indices: Sequence[int], generator=None, device=None) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        for idx in self.indices:
            yield idx

    def __len__(self) -> int:
        return len(self.indices)


class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.
        Updated class of torch to handle 'device=cuda:x' (throw error with torch.randomper otherwise)
    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
        device (string) : device to perform the sampling 
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None, device=None) -> None:
        self.indices = indices
        self.generator = generator
        self.device = device # Add, to indicate the device to torch.randperm

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator, device=self.device):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrap any sampler so that each DDP rank gets an exclusive slice of its output.
    We precompute the per‐rank indices at init (or whenever set_epoch is called),
    so you can inspect them without calling __iter__.

    Changed this class (Erwan 07/05/2025) so we can get the sub indices of the wrapper
    for each rank before calling __iter__, which seems to be the easiest way to get the 
    indices accross the processes at evaluate time.
    
    """
    def __init__(
        self,
        sampler,
        seed: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        **kwargs
    ):
        # Build the parent on the *list* of sampler indices
        super().__init__(
            list(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
            **kwargs
        )
        self.sampler = sampler

        # Placeholder for the per-rank output
        self.distributed_subsampler_indices = None
        
        # Immediately compute them for epoch 0
        self._generate_indices()

    def set_epoch(self, epoch: int):
        """
        If you use the sampler’s epoch-based shuffling, call this
        before each new epoch and re-generate the indices.
        """
        super().set_epoch(epoch)
        self._generate_indices()

    def _generate_indices(self):
        """
        Internal: run the parent's __iter__ once to get the positions,
        then apply them to your wrapped sampler's output.
        """
        # 1) get the slice positions for *this* rank
        positions = list(super().__iter__())

        # 2) fetch subsampler indices with synchronized seeding
        # By calling list(self.sampler), forces it to run through its entire __iter__ once and collect every index into a list:
        full_indices = list(self.sampler)
        
        # 3) pick out only those positions
        picked = itemgetter(*positions)(full_indices)
        
        # 4) ensure we always have a tuple
        # added it (Erwan) otherwise problem when testing with batch_size of 1 on each gpu
        if not isinstance(picked, tuple):
            picked = (picked,)

        self.distributed_subsampler_indices = picked

    def __iter__(self):
        # Just iterate over what we already computed
        return iter(self.distributed_subsampler_indices)

    def __len__(self):
        # Length should match what DistributedSampler would have reported
        return len(self.distributed_subsampler_indices)
