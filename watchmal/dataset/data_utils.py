from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from hydra.utils import instantiate
import numpy as np
from watchmal.dataset.samplers import DistributedSamplerWrapper

# TODO: remove gpu args
def get_data_loader(dataset, batch_size, sampler, num_workers, is_distributed, split_path=None, split_key=None, transforms=None):
    dataset = instantiate(dataset, transforms=transforms)
    
    if split_path is not None and split_key is not None:
        split_indices = np.load(split_path, allow_pickle=True)[split_key]
        sampler = instantiate(sampler, split_indices)
    else:
        sampler = instantiate(sampler)
    
    # TODO: move DistributedSampler functionality elsewhere
    if is_distributed:
        sampler = DistributedSamplerWrapper(sampler=sampler)
    
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)