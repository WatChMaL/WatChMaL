from operator import itemgetter
from typing import Optional

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

def SubsetSequentialSampler(indices):
    return indices


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        seed,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true,
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            list(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        self.sampler = sampler
        self.epoch = 0
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # fetch DistributedSampler indices
        indexes_of_indexes = super().__iter__()
        
        # deterministically shuffle based on epoch
        updated_seed = self.seed + int(self.epoch)
        torch.manual_seed(updated_seed)

        # fetch subsampler indices with synchronized seeding
        subsampler_indices = list(self.sampler)

        # get subsampler_indexes[indexes_of_indexes]
        distributed_subsampler_indices = itemgetter(*indexes_of_indexes)(subsampler_indices)

        return iter(distributed_subsampler_indices)