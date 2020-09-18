from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from hydra.utils import instantiate
import numpy as np

# TODO: remove this class, and replace with sequentialsampler
class SubsetSequenceSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self,indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class DataModule():

    def __init__(self, dataset, train_batch_size, val_batch_size, split_path, num_workers):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset_config = dataset
        split_indices = np.load(split_path, allow_pickle=True)
        self.train_indices = split_indices["train_idxs"]
        self.val_indices = split_indices["val_idxs"]
        self.test_indices = split_indices["test_idxs"]
        self.num_workers = num_workers
        
        self.dataset = instantiate(self.dataset_config)
        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetSequenceSampler(self.test_indices)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.train_batch_size, sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.val_batch_size, sampler=self.val_sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.val_batch_size, sampler=self.test_sampler, num_workers=self.num_workers)