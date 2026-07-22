"""
PyG-oriented dataset for hierarchical (PMT + mPMT) graphs.
Wraps two InMemoryDatasets (20-inch and mPMT level) and returns a single Data
per event with PMT graph, optional mPMT graph attached as .mPMT_data, and
mPMT mapping attributes (pmt_to_mPMT, num_mPMTs, etc.).
"""

import torch
from torch_geometric.data import Batch, InMemoryDataset

from typing import Callable

from watchmal.dataset.graph.pyg_in_memory_20inch_pmt import PyGInMemory20inchDataset
from watchmal.utils.logging_utils_caverns import setup_logging


log = setup_logging(__name__)

def collate_fn(batch):
    pmt_batch = Batch.from_data_list(batch)
    labels = pmt_batch.y if hasattr(pmt_batch, "y") else None

    has_mpmt = (
        batch
        and hasattr(batch[0], "mPMT_data")
        and getattr(batch[0], "mPMT_data", None) is not None
    )
    if has_mpmt:
        mpmt_batch = Batch.from_data_list([d.mPMT_data for d in batch])
        return {
            "pmt_data": {"data": pmt_batch, "target": labels},
            "mpmt_data": mpmt_batch,
            "labels": labels,
        }
    return {
        "data": pmt_batch,
        "labels": labels,
    }

class PyGInMemoryMPMTDataset(InMemoryDataset):
    """
    Dataset wrapper that provides both PMT (20-inch) and mPMT level graphs
    as a single PyG Data per event.
    It is not possible to instantiate directly the two datasets here, because self.load() 
    from PyG would overwrite one of the two datasets. Hence we use 2 instances of PyGInMemory20inchDataset.
    """

    def __init__(
        self,
        pyg_data_folder_path,
        pyg_data_file_names: list[str],
        transforms: list[Callable] | None = None,
        mPMT_transforms: list[Callable] | None = None,
    ):
        super().__init__(root=pyg_data_folder_path, transform=None)

        assert len(pyg_data_file_names) == 2, (
            f"PyG data file names must be a list of two pmt_data.pt and mpmt_data.pt, "
            f"got {pyg_data_file_names}"
        )

        log.info(f"Len [before load]: {self.len()}")
        log.info(f"Loading from : {self.processed_dir}")
        self.pmt_dataset = PyGInMemory20inchDataset(pyg_data_folder_path, [pyg_data_file_names[0]], transforms)
        self.mPMT_dataset = PyGInMemory20inchDataset(pyg_data_folder_path, [pyg_data_file_names[1]], mPMT_transforms)
        log.info(f"Len [after load]: {self.len()}")
        
        assert len(self.pmt_dataset) == len(self.mPMT_dataset), (
                f"PMT and mPMT datasets must have same length, "
                f"got {len(self.pmt_dataset)} and {len(self.mPMT_dataset)}"
            )

        self.transforms = transforms
        self.mPMT_transforms = mPMT_transforms
        log.info(f"transforms : {self.transforms}")
        log.info(f"mPMT_transforms : {self.mPMT_transforms}")


    def __len__(self):
        return len(self.pmt_dataset)

    def get(self, idx):
        """Return a single PyG Data: PMT graph with transforms, mapping, and optional .mPMT_data."""
        pmt_data = self.pmt_dataset[idx]
        mPMT_data = self.mPMT_dataset[idx]

        return {'pmt_data': pmt_data, 'mpmt_data': mPMT_data}

    def __getitem__(self, idx):
        return self.get(idx)

    def _dataset_collate(self):
        """
        Return the collate function to pass to torch.utils.data.DataLoader
        when using this dataset. Batches the list of Data (each with optional
        .mPMT_data) and returns a format the engine expects:
        - If samples have .mPMT_data: {"pmt_data": {"data": pmt_batch, "target": y},
          "mpmt_data": mpmt_batch} so the engine's mPMT branch sets
          self.data = {"pmt": pmt_batch, "mpmt": mpmt_batch} for HierarchicalGAT.
        - Else: {"data": pmt_batch, "labels": y} for the vanilla branch.
        """
        return collate_fn
