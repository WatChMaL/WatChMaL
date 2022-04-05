import torch_geometric.data as PyGData

import h5py
import numpy as np
import torch

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
import watchmal.dataset.graph_io_utils as du


##...............................................

class GNN_dataset(H5Dataset, PyGData.Dataset):

    def __init__(self, h5file, geometry_file, is_distributed=False, transforms=None):
        super().__init__(h5file, is_distributed)

        geo_file = np.load(geometry_file, 'r')
        self.geo_positions = torch.from_numpy(geo_file['position']).float()
        self.geo_orientations = torch.from_numpy(geo_file['orientation']).float()
        
        self.f = h5py.File(h5file,'r')
        h5_labels = self.f['labels']
        self.labels = np.array(h5_labels)

    def load_edges(self, nhits):
        edge_index = torch.ones([nhits, nhits], dtype=torch.int64)
        edge_index = edge_index * (1 - torch.eye(nhits, nhits))
        self.edge_index=edge_index.to_sparse()._indices()

    def  __getitem__(self, item):
        data_dict = super().__getitem__(item)
        hit_pmts = data_dict['data']['hit_pmts']
        hit_charges = data_dict['data']['hit_charges']
        hit_times = data_dict['data']['hit_times']

        hit_positions = self.geo_positions[hit_pmts, :]
        n_hits = hit_pmts.shape[0]
            
        hit_orientations = self.geo_orientations[hit_pmts, :]
        data = np.zeros((8, n_hits))
        data[3:6, :n_hits] = hit_orientations.T
        data[:3, :n_hits] = hit_positions[:n_hits].T
        data[-2, :n_hits] = hit_charges[:n_hits]
        data[-1, :n_hits] = hit_times[:n_hits]
        
        self.load_edges(n_hits)
        y = torch.tensor([self.labels[item]], dtype=torch.int64)
        
        data = np.transpose(data)
        
        x = torch.from_numpy(data).to(torch.float)
        
        return PyGData.Data(x=x, y=y, edge_index=self.edge_index)
    
    def __len__(self):
        return self.labels.shape[0]        
