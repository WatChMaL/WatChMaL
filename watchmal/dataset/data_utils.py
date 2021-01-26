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
    """
    Returns data loaders given dataset and sampler configs

    Args:
        dataset         ... hydra config specifying dataset object
        batch_size      ... batch size
        sampler         ... hydra config specifying sampler object
        num_workers     ... number of workers to use in dataloading
        is_distributed  ... whether running in multiprocessing mode, used to wrap sampler using DistributedSamplerWrapper
        seed            ... seed used to coordinate samplers in distributed mode
        split_path      ... path to indices specifying splitting of dataset among train/val/test
        split_key       ... string key to select indices
        transforms      ... list of transforms to apply
    
    Returns: dataloader created with instantiated dataset and (possibly wrapped) sampler
    """
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
    