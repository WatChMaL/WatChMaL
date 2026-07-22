"""
PyG-specific dataset factory.
Centralizes construction of graph datasets from config (dataset_parameters + transforms).
Uses Hydra instantiate - config should contain _target_ pointing to the dataset class.
Multiple folder paths → one dataset per folder, concatenated with PyGConcatDataset.
"""
import random
from omegaconf import DictConfig, ListConfig, OmegaConf

# hydra imports
from hydra.utils import instantiate

# pyg imports
import torch
import torch_geometric.transforms as T

# watchmal imports
from watchmal.dataset.graph.pyg_concat import PyGConcatDataset
from watchmal.utils.logging_utils_caverns import setup_logging

log = setup_logging(__name__)


def get_dataset(
    data_config: DictConfig | dict,
    mPMT_transforms_config: DictConfig | dict | None = None,  # For future
):
    """
    Build a PyG dataset from full data config.
    
    Extracts dataset_config, transforms_config, and dataset_parameters from the input.
    The dataset_parameters config should contain _target_ pointing to the dataset class.
    Multiple folder paths → one dataset per folder, concatenated with PyGConcatDataset.

    Variable names match the dataset class inits: pyg_data_folder_path, pyg_data_file_names.
    Backward compat: "folder_path" / "graph_folder_path" and "graph_file_names" also accepted.

    Parameters
    ----------
    data_config : DictConfig or dict
        Full data config containing:
        - data.dataset or dataset: dataset config with _target_, dataset_parameters, etc.
        - data.transforms or transforms: transforms config
        
        dataset_parameters must contain:
        - _target_ : str (path to dataset class, e.g., "watchmal.dataset.graph.pyg_in_memory_20inch_pmt.PyGInMemory20inchDataset")
        - pyg_data_folder_path : str or list[str] (or "folder_path" / "graph_folder_path")
        - pyg_data_file_names : list[str] (or "graph_file_names")
          Length 1 → 20-inch, length 2 → MPMT.
    mPMT_transforms_config : DictConfig or dict, optional
        Config for mPMT transforms (not yet supported).

    Returns
    -------
    PyGInMemory20inchDataset, PyGInMemoryMPMTDataset, or PyGConcatDataset
    """
    # Extract dataset and transforms configs (handle both data.dataset and dataset formats)
    if hasattr(data_config, 'data'):
        dataset_config = data_config.data.dataset
        transforms_config = data_config.data.transforms
    elif 'data' in data_config:
        dataset_config = data_config['data']['dataset']
        transforms_config = data_config['data']['transforms']
    else:
        # Legacy: assume data_config IS the dataset config
        dataset_config = data_config.get('dataset', data_config)
        transforms_config = data_config.get('transforms', None)
    
    dataset_parameters = dataset_config.dataset_parameters if hasattr(dataset_config, 'dataset_parameters') else dataset_config['dataset_parameters']
    dict_config = OmegaConf.to_container(dataset_parameters, resolve=True) if isinstance(dataset_parameters, DictConfig) else dict(dataset_parameters)
    
    # Extract folder path(s) with backward compat (using pop like reference file)
    base_pyg_folder = dict_config.pop("pyg_data_folder_path", None) or dict_config.pop("folder_path", None) or dict_config.pop("graph_folder_path", None)
    pyg_data_file_names = dict_config.pop("pyg_data_file_names", None) or dict_config.pop("graph_file_names", None)
    
    if base_pyg_folder is None:
        raise ValueError("dataset_parameters must contain 'pyg_data_folder_path', 'folder_path', or 'graph_folder_path'")
    if pyg_data_file_names is None or len(pyg_data_file_names) == 0:
        raise ValueError("dataset_parameters must contain 'pyg_data_file_names' or 'graph_file_names' (non-empty list)")
    
    # Normalize folder paths to list (single str → one-element list)
    folder_paths = list(base_pyg_folder) if isinstance(base_pyg_folder, (list, ListConfig)) else [base_pyg_folder]
    
    n_files = len(pyg_data_file_names)
    assert n_files in (1, 2), f"pyg_data_file_names must have length 1 (20-inch) or 2 (MPMT), got {n_files}"
    assert len(folder_paths) > 0, "At least one folder path is required"

    # Instantiate transforms
    transform_compose = None
    if transforms_config is not None and 'transforms' in transforms_config.keys():
        transform_list = []
        for _, trf_config in transforms_config['transforms'].items():
            transform = instantiate(trf_config)
            transform_list.append(transform)
        transform_compose = T.Compose(transform_list)

    # mPMT transforms (not yet supported)
    # mPMT_transform_compose = None
    # if mPMT_transforms_config is not None and 'transforms' in mPMT_transforms_config.keys():
    #     mPMT_transform_list = []
    #     for _, trf_config in mPMT_transforms_config['transforms'].items():
    #         transform = instantiate(trf_config)
    #         mPMT_transform_list.append(transform)
    #     mPMT_transform_compose = T.Compose(mPMT_transform_list)

    ## Instantiating the dataset(s)
    if len(folder_paths) == 1:
        log.info(f"One pyg folder path, not using PyGConcatDataset")
        dataset = instantiate(
            dict_config,
            pyg_data_folder_path=folder_paths[0],
            pyg_data_file_names=pyg_data_file_names,
            transforms=transform_compose,
        )
        return dataset

    # Multiple folders: instantiate one dataset per folder, then concat
    log.info(f"Multiple folder paths ({len(folder_paths)}): building one dataset per folder and using PyGConcatDataset")
    all_datasets = []
    for folder_path in folder_paths:
        dataset = instantiate(
            dict_config,
            pyg_data_folder_path=folder_path,
            pyg_data_file_names=pyg_data_file_names,
            transforms=transform_compose,
        )
        all_datasets.append(dataset)
    
    final_dataset = PyGConcatDataset(all_datasets)
    log.info(f"Dataset loaded. Length: {len(final_dataset)}. First sample: {final_dataset[0]}")
    return final_dataset


def match_type(obj_type: str):

    match obj_type:
        case 'int16':
            to_type = torch.int16
        case 'int32':
            to_type = torch.int32
        case 'int64':
            to_type = torch.int64
        case 'float16':
            to_type = torch.float16
        case 'float32':
            to_type = torch.float32
        case 'float64':
            to_type = torch.float64
        case _:
            log.info(f"match_type : Value Error, to_type {to_type} is not supported")
            log.info("Add the data type into the functionn or change the new target type\n\n")
            raise ValueError

    return to_type


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
