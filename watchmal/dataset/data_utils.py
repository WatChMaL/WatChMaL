"""
Utils for handling creation of dataloaders
"""

# hydra imports

# torch imports
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

# generic imports
import numpy as np
import random

# WatChMaL imports
from WatChMaL.watchmal.dataset.samplers import DistributedSamplerWrapper

def get_data_loader(settings, dataset, batch_size, sampler, num_workers, is_distributed, seed, indices=0, split_path=None, split_key=None, transforms=None, data_class=None, use_positions=True, use_time=True, pmt_positions_file=None):
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
    #PointNet
    if 'PointNet'.lower() in settings.arch.lower():
        dataset = data_class(dataset, use_positions=use_positions, use_times=use_time, is_distributed=is_distributed)
    #ResNet
    if 'ResNet'.lower() in settings.arch.lower():
        dataset = data_class(dataset, pmt_positions_file, is_distributed=is_distributed)
    #dataset = instantiate(dataset, transforms=transforms, is_distributed=is_distributed)
    
    if split_path is not None and split_key is not None:
        split_indices = np.load(split_path, allow_pickle=True)[split_key]
        sampler = instantiate(sampler, split_indices)
    else:
        sampler = sampler(indices)
    
    if is_distributed:
        ngpus = torch.distributed.get_world_size()

        batch_size = int(batch_size/ngpus)
        
        sampler = DistributedSamplerWrapper(sampler=sampler, seed=seed)
    
    # TODO: added drop_last, should decide if we want to keep this
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, drop_last=False, persistent_workers=True, pin_memory=True)


def get_transformations(transformations, transform_names):
    if transform_names is not None:
        for transform_name in transform_names:
            assert hasattr(transformations, transform_name), f"Error: There is no defined transform named {transform_name}"
        transform_funcs = [getattr(transformations, transform_name) for transform_name in transform_names]
        return transform_funcs
    else:
        return None


def apply_random_transformations(transforms, data, segmented_labels = None):
    if transforms is not None:
        for transformation in transforms:
            if random.getrandbits(1):
                data = transformation(data)
                if segmented_labels is not None:
                    segmented_labels = transformation(segmented_labels)
    return data