

from typing import Callable
from torch_geometric.data import InMemoryDataset

# watchmal imports
from watchmal.utils.logging_utils_caverns import setup_logging

log = setup_logging(__name__)


class PyGInMemory20inchDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for 20-inch (PMT-level) graphs only.
    Decision not to use the transform from pyg bc not convenient for storing
    transformed data in the cache.
    """

    def __init__(self, 
        pyg_data_folder_path: str,
        pyg_data_file_names: list[str] = ['data.pt'],
        transforms: list[Callable] | None = None
    ):
        super().__init__(root=pyg_data_folder_path, transform=None)

        assert len(pyg_data_file_names) == 1, (
            f"If using PyGInMemory20inchDataset, pyg_data_file_names must be a list of one data.pt, "
            f"got {pyg_data_file_names}"
        )
        self.pyg_data_file_names = pyg_data_file_names
        self.transforms = transforms

        log.info(f"transforms : {self.transforms}")
        log.info(f"Processed path : {self.processed_paths}")
        log.info(f"Processed dir : {self.processed_dir}")

        log.info(f"Len [before load]: {self.len()}")
        log.info(f"Loading from : {self.processed_dir}/{self.processed_file_names[0]}")
        self.load(f"{self.processed_dir}/{self.processed_file_names[0]}")
        log.info(f"Len [after load]: {self.len()}")

    @property
    def processed_file_names(self):
        return self.pyg_data_file_names


    def get(self, idx):
    # Caution : 
    # The pipeline for get is : (starting at the loader)
    # getitem (Dataset) -> get (this class) 
    # If you can super().get() in this class, it's the one from
    # InMemoryDataset, where the big Data is sliced in Data cached in _data_list.
    # then Dataset gets this data, and apply transform.

    # Conclusion : If transform is given in __init__(),
    # this data object will NOT have already be transformed.
    # (See torch_geometric.Data.Dataset.__getitem__() l.292 - 293)
        data = super().get(idx)
            
        # debug purpose
        # log.info(f"Graph [before idx & transforms] : {data}")
        # log.info(f"Data.y, idx: {idx} : {data.y}")
        # for node in range(3):
        #     log.info(f"Data.x, idx: {idx}, node: {node} : {data.x[node]}")
            
        data.idx = idx # for watchmal compatibility
        data = data if self.transforms is None else self.transforms(data.clone())

        # debug purpose
        # log.info(f"Graph [after idx & transforms] : {data}")

        # if isinstance(data, dict):
        #     log.info(f"Data.y, idx: {idx} : {data['data'].y}")
        #     for node in range(3):
        #         log.info(f"Data.x, idx: {idx}, node: {node} : {data['data'].x[node]}")
        # else:
        #     log.info(f"Data.y, idx: {idx} : {data.y}")
        #     for node in range(3):
        #         log.info(f"Data.x, idx: {idx}, node: {node} : {data.x[node]}")

        return data


    def map_labels(in_label, label_set):
        # This method is for watchmal compatibility
        pass


# Backward compatibility (i'm lazy)
EasyInMemoryDataset = PyGInMemory20inchDataset