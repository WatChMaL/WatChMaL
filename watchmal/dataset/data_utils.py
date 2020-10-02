from torch.utils.data import DataLoader
from hydra.utils import instantiate
import numpy as np


def subset_sequential_sampler(indices):
    return indices


def get_data_loader(dataset, batch_size, sampler, num_workers=0, split_path=None, split_key=None, transforms=None):
    print("Call to get_data_loader")
    print(dataset)
    dataset = instantiate(dataset, transforms=transforms)
    print("Call to get_data_loader 2")
    if split_path is not None and split_key is not None:
        split_indices = np.load(split_path, allow_pickle=True)[split_key]
        sampler = instantiate(sampler, split_indices)
    else:
        sampler = instantiate(sampler)
    print("Call to get_data_loader 3")
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)