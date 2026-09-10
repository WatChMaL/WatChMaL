"""
Utils for handling creation of dataloaders
"""

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch.utils.data import DataLoader

# generic imports
import logging
import numpy as np
import random

# WatChMaL imports
from watchmal.dataset.samplers import DistributedSamplerWrapper

# pyg imports
from torch_geometric.loader import DataLoader as PyGDataLoader

log = logging.getLogger(__name__)


def get_data_loader(dataset, batch_size, sampler, num_workers, is_distributed, is_gpu, seed, is_graph=False,
                    split_path=None, split_key=None, pre_transforms=None, post_transforms=None, drop_last=False,
                    loader_name=None):
    """
    Creates a dataloader given the dataset and sampler configs. The dataset and sampler are instantiated using their
    corresponding configs. If using DistributedDataParallel, the sampler is wrapped using DistributedSamplerWrapper.
    A dataloader is returned after being instantiated using this dataset and sampler.


    Parameters
    ----------
    dataset
        Hydra config specifying dataset object.
    batch_size : int
        Size of the batches that the data loader should return.
    sampler
        Hydra config specifying sampler object.
    num_workers : int
        Number of data loader worker processes to use.
    is_distributed : bool
        Whether running in multiprocessing mode (i.e. DistributedDataParallel)
    seed : int
        Random seed used to coordinate samplers in distributed mode.
    is_graph : bool
        A boolean indicating whether the dataset is graph or not, to use PyTorch Geometric data loader if it is graph. False by default.
    split_path
        Path to an npz file containing an array of indices to use as a subset of the full dataset.
    split_key : string
        Name of the array to use in the file specified by split_path.
    pre_transforms : list of string
        List of transforms to apply to the dataset before any transforms specified by the dataset config.
    pre_transforms : list of string
        List of transforms to apply to the dataset after any transforms specified by the dataset config.
    drop_last : bool
        Whether to drop the last incomplete batch of each epoch. False by default, but forced to True for the
        training loader when the last batch would hold a single event (see below).
    loader_name : string
        Name of the loader in the task config (e.g. "train", "validation", "test"). Used to identify the training
        loader, which is the only one that needs the single-item batch guard below.

    Returns
    -------
    torch.utils.data.DataLoader
        dataloader created with instantiated dataset and (possibly wrapped) sampler
    """
    # combine transforms specified in data loader with transforms specified in dataset
    transforms = dataset["transforms"] if (("transforms" in dataset) and (dataset["transforms"] is not None)) else []
    transforms = (pre_transforms or []) + transforms + (post_transforms or [])
    dataset = instantiate(dataset, transforms=(transforms or None))
    
    if split_path is not None and split_key is not None:
        split_indices = np.load(split_path, allow_pickle=True)[split_key]
        sampler = instantiate(sampler, split_indices)
    else:
        sampler = instantiate(sampler)
    
    if is_distributed:
        ngpus = torch.distributed.get_world_size()

        batch_size = int(batch_size/ngpus)
        
        sampler = DistributedSamplerWrapper(sampler=sampler, seed=seed)

    # A final batch holding exactly one event breaks BatchNorm in training mode, which raises
    # "Expected more than 1 value per channel when training", and its SyncBatchNorm conversion likewise.
    # Whether that happens depends on the per-rank sample count and the per-rank batch size, and neither of
    # those is a number written in a config file: both are derived here, after the sampler has been wrapped
    # for DistributedDataParallel and the batch size divided by the world size. Setting drop_last in the
    # config therefore cannot express the condition, so it is checked here, where both numbers are known.
    # The training loader is the only one that needs it: validate() and evaluate() run under model.eval(),
    # where BatchNorm reads its running statistics instead of the batch, and dropping an event there would
    # silently shorten the evaluation output.
    if loader_name == "train":
        try:
            per_rank_count = len(sampler)
        except TypeError:  # a sampler that does not report a length
            per_rank_count = None
        if batch_size == 1:
            log.warning(f"Data loader '{loader_name}': batch_size is 1 per rank, so every batch holds a single "
                        f"event and BatchNorm layers will fail in training mode; drop_last cannot help here.")
        elif per_rank_count is not None and not drop_last and per_rank_count % batch_size == 1:
            log.warning(f"Data loader '{loader_name}': per_rank_count ({per_rank_count}) % batch_size "
                        f"({batch_size}) == 1, and that trailing single-event batch would break SyncBatchNorm; "
                        f"forcing drop_last=True on this loader, dropping 1 event per rank per epoch.")
            drop_last = True

    if is_graph:
        return PyGDataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers,
                             drop_last=drop_last)
    else:
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last,
                          persistent_workers=(num_workers > 0), pin_memory=is_gpu, multiprocessing_context='fork')


def get_transformations(transformations, transform_names):
    """
    Returns a list of transformation functions from an object and a list of names of the desired transformations, where
    the object has functions with the given names.

    Parameters
    ----------
    transformations : object containing the transformation functions
    transform_names : list of strings

    Returns
    -------

    """
    if transform_names is not None:
        for transform_name in transform_names:
            assert hasattr(transformations, transform_name), f"Error: There is no defined transform named {transform_name}"
        transform_funcs = [getattr(transformations, transform_name) for transform_name in transform_names]
        return transform_funcs
    else:
        return None


def apply_random_transformations(transforms, data, segmented_labels=None):
    """
    Randomly chooses a set of transformations to apply, from a given list of transformations, then applies those that
    were randomly chosen to the data and returns the transformed data.

    Parameters
    ----------
    transforms : list of callable
        List of transformation functions to apply to the data.
    data : array_like
        Data to transform
    segmented_labels
        Truth data in the same format as data, to also apply the same transformation.

    Returns
    -------
    data
        The transformed data.
    """
    if transforms is not None:
        for transformation in transforms:
            if random.getrandbits(1):
                data = transformation(data)
                if segmented_labels is not None:
                    segmented_labels = transformation(segmented_labels)
    return data
