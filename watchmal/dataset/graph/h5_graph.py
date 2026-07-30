import os.path as osp
import h5py
import torch
import numpy as np

from torch_geometric.data import Dataset, Data
from torch_geometric.nn import radius_graph, knn_graph

class H5GraphDataset(Dataset):
    def __init__(self, pyg_data_folder_path, pyg_data_file_names=('data.h5',),
                 transforms=None, k=None, radius=None,
                 time_order=False, verbose=0):
        super().__init__(root=pyg_data_folder_path, transform=None)

        self.transforms = transforms
        self.h5_paths = [osp.join(self.processed_dir, fn) for fn in list(pyg_data_file_names)]
        self.radius = radius
        self.k = k
        self.verbose = verbose
        self.time_order = time_order
        self.file_event_counts = []
        for path in self.h5_paths:
            with h5py.File(path, "r") as f:
                self.file_event_counts.append(len(f["node_offsets"]))
        
        self.num_graphs = sum(self.file_event_counts)
        self.files = None

    @property
    def processed_dir(self):
        proc_dir = osp.join(self.root, "processed")
        return proc_dir if osp.exists(proc_dir) else self.root

    def _open_files(self):
        if self.files is None:
            self.files = [
                h5py.File(
                    p,
                    "r",
                    swmr=True,
                    libver="latest",
                    rdcc_nbytes=1024**2,
                    rdcc_nslots=1_000,
                    rdcc_w0=0.0,
                )
                for p in self.h5_paths
            ]

    def __len__(self):
        return self.num_graphs

    def __getstate__(self):
        # h5py file handles can't be pickled — DataLoader workers open their own copy lazily.
        state = self.__dict__.copy()
        state['files'] = None
        return state

    def _map_index(self, idx):
        running = 0
        for file_idx, count in enumerate(self.file_event_counts):
            if idx < running + count:
                return file_idx, idx - running
            running += count
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx):
        self._open_files()
        
        file_idx, local_idx = self._map_index(idx)
        f = self.files[file_idx]

        # 1. Fetch Node/Edge Ranges
        node_end = f["node_offsets"][local_idx]
        node_start = 0 if local_idx == 0 else f["node_offsets"][local_idx - 1]
        edge_end = f["edge_offsets"][local_idx]
        edge_start = 0 if local_idx == 0 else f["edge_offsets"][local_idx - 1]

        # 2. Load hit PMTs and features
        x = torch.from_numpy(f["x"][node_start:node_end]).to(torch.float32)
        y = torch.tensor(f["y"][local_idx])

        # --- FALLBACK LOGIC FOR PMT_ID ---
        # Check if "pmt_id" exists in the current HDF5 file
        if "pmt_id" in f:
            pmt_id = torch.from_numpy(f["pmt_id"][node_start:node_end]).to(torch.long)
        else:
            # Fallback option: set to None (PyG allows custom attributes to be None)
            # Alternatively, use torch.zeros(x.size(0), dtype=torch.long) if you need a placeholder
            pmt_id = None
        # ---------------------------------

        edge_index = torch.from_numpy(f["edge_index"][:, edge_start:edge_end]).to(torch.long)

        # 3. Dynamic Graph Construction
        if self.k is not None and self.radius is None:
            edge_index = knn_graph(x[:, -3:], k=self.k, loop=False)
            
        if self.radius is not None:
            edge_index = radius_graph(
                x[:, -3:], 
                r=self.radius, 
                loop=False, 
                max_num_neighbors=self.k if self.k is not None else 64
            )

        # 4. Time Ordering
        if self.time_order:
            row, col = edge_index
            time = x[:, 0]
            mask = time[col] <= time[row]
            edge_index = edge_index[:, mask].clone()

        # 5. Build Data Object
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Only attach pmt_id if it was successfully found
        if pmt_id is not None:
            data.pmt_id = pmt_id
            
        data.idx = torch.tensor([idx], dtype=torch.long)

        # 6. Apply Transforms
        if self.transforms is not None:
            data = self.transforms(data)

        return data