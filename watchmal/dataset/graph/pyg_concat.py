


from typing import List, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from watchmal.utils.logging_utils_caverns import setup_logging

log = setup_logging(__name__)

class PyGConcatDataset(Dataset):

    def __init__(self, datasets: List[Dataset]):
        self._datasets = datasets
        self._len = sum(len(dataset) for dataset in datasets)
        self._indexes = []

        # Calculate distribution of indexes in all datasets
        cumulative_index = 0
        for idx, dataset in enumerate(datasets):
            next_cumulative_index = cumulative_index + len(dataset)
            self._indexes.append((cumulative_index, next_cumulative_index, idx))
            cumulative_index = next_cumulative_index

        log.info(f"[PygConcatDataset] Datasets summary length: {self._len}")
        log.info(f"[PygConcatDataset] Datasets indexes: {self._indexes}")

    def __getitem__(self, index) -> Union[Tuple, List[Tuple]]:
       
        """Handle both integer and slice indexing"""
        if isinstance(index, slice):
            # Generate indices from slice
            indices = range(*index.indices(len(self)))
            return [self[i] for i in indices]
        
        elif isinstance(index, (int, np.integer)):
            # Handle negative indexing
            if index < 0:
                index += len(self)
            
            for start, stop, dataset_index in self._indexes:
                if start <= index < stop:
                    return self._datasets[dataset_index][index - start]
            raise IndexError(f"Index {index} out of range")
        
        else:
            raise TypeError(f"Invalid index type {type(index)}")

    def __len__(self) -> int:
        return self._len

    @property
    def processed_dir(self):
        return [dataset.processed_dir for dataset in self._datasets]

    @property
    def processed_file_names(self, i=0):
        return [dataset.processed_file_names[i] for dataset in self._datasets]

    @property
    def transforms(self):
        # We consider all transforms are the same across all datasets, hence [0]
        return self._datasets[0].transforms

    def _dataset_collate(self):
        """
        Return the collate function for the DataLoader when using this concat dataset.
        Delegates to the first sub-dataset's _dataset_collate (e.g. for hierarchical
        PMT+mPMT batches); all sub-datasets are assumed to use the same collate.
        """
        if self._datasets and hasattr(self._datasets[0], "_dataset_collate") and callable(getattr(self._datasets[0], "_dataset_collate")):
            return self._datasets[0]._dataset_collate()
        return None
