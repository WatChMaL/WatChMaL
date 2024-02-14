"""
Utils for handling creation of dataloaders
"""

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch.utils.data import DataLoader

# generic imports
import numpy as np
import random

# WatChMaL imports
from watchmal.dataset.samplers import DistributedSamplerWrapper

import torchvision

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

# pyg imports
#from torch_geometric.loader import DataLoader as PyGDataLoader


def get_data_loader(dataset, batch_size, sampler, num_workers, is_distributed, seed, is_graph=False,
                    split_path=None, split_key=None, pre_transforms=None, post_transforms=None):
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

    if is_graph:
        return PyGDataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    else:
        # TODO: added drop_last, should decide if we want to keep this
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, drop_last=False,
                          persistent_workers=(num_workers > 0), pin_memory=True)


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


def apply_random_transformations(transforms, data, segmented_labels=None, counter=0):
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
            if "rotate_cylinder" in transformation.__name__:
                #save_fig(data[0],False, counter=counter)
                data, displacement = transformation(torch.Tensor.numpy(data))
                #save_fig((torch.Tensor.numpy(data))[0],True, counter=counter)
            else:
                continue
        for transformation in transforms:
            if "rotate_cylinder" in transformation.__name__:
                continue
            if random.getrandbits(1):
                data = transformation(data)
                if segmented_labels is not None:
                    segmented_labels = transformation(segmented_labels)
    return data

def save_fig(data,isPost, displacement=0, counter=0):
    print("SAVE FIG")
    print(data.size())
    plt.imshow(data.numpy(), interpolation='none')
    print("1")
    cbar = plt.colorbar()
    print("2")
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("PMT Charge", rotation=270)
    plt.xlabel('X pixels')
    plt.ylabel('Y pixels')
    if isPost:
        plt.savefig('/home/fcormier/t2k/ml/t2k_ml_training/plots/'+str(counter)+'_post_rot_dc_img_dis'+str(displacement)+'.png')
    else:
        plt.savefig('/home/fcormier/t2k/ml/t2k_ml_training/plots/'+str(counter)+'_pre_rot_img'+'.png')
    plt.clf()
    print("3")

