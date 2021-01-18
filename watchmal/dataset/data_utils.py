"""
Utils for handling creation of dataloaders
"""

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

# generic imports
import numpy as np

# WatChMaL imports
from watchmal.dataset.samplers import DistributedSamplerWrapper

def get_data_loader(dataset, batch_size, sampler, num_workers, is_distributed, seed, split_path=None, split_key=None, transforms=None):
    dataset = instantiate(dataset, transforms=transforms, is_distributed=is_distributed)
    
    if split_path is not None and split_key is not None:
        split_indices = np.load(split_path, allow_pickle=True)[split_key]
        sampler = instantiate(sampler, split_indices)
    else:
        sampler = instantiate(sampler)
    
    if is_distributed:
        ngpus = torch.distributed.get_world_size()

        batch_size = int(batch_size/ngpus)
        
        sampler = DistributedSamplerWrapper(sampler=sampler, seed=seed)
    
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    