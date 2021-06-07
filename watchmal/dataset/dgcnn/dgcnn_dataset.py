import torch_geometric.data as tgd

import h5py
import numpy as np
import torch

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du


##...............................................

class DGCNN_dataset(H5Dataset, tgd.Dataset):

    def __init__(self, h5file, geometry_file, is_distributed, k_neighbours, dynamic=True, n_points=4000, use_orientations=True, transforms=None, is_graph=True):
        super().__init__(h5file, is_distributed)
        
        self.n_points = n_points
        geo_file = np.load(geometry_file, 'r')
        self.geo_positions = torch.from_numpy(geo_file["position"]).float()
        self.geo_orientations = torch.from_numpy(geo_file["orientation"]).float()
        
        self.f = h5py.File(h5file,'r')
        h5_labels = self.f["labels"]
        self.labels = np.array(h5_labels)
        self.dynamic = dynamic
        self.k_neighbours = k_neighbours
        self.use_orientations = use_orientations
       
    
    def load_edges(self, idx, nhits):
        edge_index = torch.ones([nhits, nhits], dtype=torch.int64)
        self.edge_index=edge_index.to_sparse()._indices()
        

    def  __getitem__(self, item):

        data_dict = super().__getitem__(item)
        hit_pmts = data_dict["data"]["hit_pmts"]
        hit_charges = data_dict["data"]["hit_charges"]
        hit_times = data_dict["data"]["hit_times"]

        hit_positions = self.geo_positions[hit_pmts, :]
        n_hits = min(self.n_points, hit_pmts.shape[0])
        
        if self.dynamic:
            self.n_points = n_hits            #dynamic graph
        else:
            self.n_points = 600               #static graph
            
        if not self.use_orientations:
            data = np.zeros((5, self.n_points))
        else:
            hit_orientations = self.geo_orientations[hit_pmts[:n_hits], :]
            data = np.zeros((7, self.n_points))
            data[3:6, :n_hits] = hit_orientations.T
        data[:3, :n_hits] = hit_positions[:n_hits].T
        data[-2, :n_hits] = hit_charges[:n_hits]
        data[-1, :n_hits] = hit_times[:n_hits]
        
        self.load_edges(item, n_hits)
        y = torch.tensor([self.labels[item]], dtype=torch.int64)
        
        data = np.transpose(data)
        
        x = torch.from_numpy(data).to(torch.float)
        return tgd.Data(x=x, y=y, edge_index=self.edge_index)
        
    