from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from hydra.utils import instantiate
import numpy as np


def subset_sequential_sampler(indices):
    return indices

# TODO: remove gpu args
def get_data_loader(dataset, batch_size, sampler, num_workers, split_path=None, split_key=None, transforms=None, gpu=None, ngpus=1):
    dataset = instantiate(dataset, transforms=transforms)
    if split_path is not None and split_key is not None:
        split_indices = np.load(split_path, allow_pickle=True)[split_key]
        sampler = instantiate(sampler, split_indices)
    else:
        sampler = instantiate(sampler)
    # TODO: move DistributedSampler functionality elsewhere
    if ngpus > 1:
        print("Using distributed sampler")
        sampler = DistributedSampler(dataset,
                                        num_replicas=ngpus,
                                        rank=gpu)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)