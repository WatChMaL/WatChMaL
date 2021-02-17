"""
Class implementing a mPMT dataset for pointnet in h5 format
"""

# torch imports
import torch

# generic imports
import numpy as np

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du

class PointNetDataset(H5Dataset):

    def __init__(self, h5file, geometry_file, is_distributed, use_orientations=False, n_points=4000, transforms=None):
        super().__init__(h5file, is_distributed)
        geo_file = np.load(geometry_file, 'r')
        self.geo_positions = torch.from_numpy(geo_file["position"]).float()
        self.geo_orientations = torch.from_numpy(geo_file["orientation"]).float()
        self.use_orientations = use_orientations
        self.n_points = n_points
        self.transforms = du.get_transformations(transformations, transforms)


    def  __getitem__(self, item):

        data_dict = super().__getitem__(item)
        hit_pmts = data_dict["data"]["hit_pmts"]
        hit_charges = data_dict["data"]["hit_charges"]
        hit_times = data_dict["data"]["hit_times"]

        hit_positions = self.geo_positions[hit_pmts, :]
        n_hits = min(self.n_points, hit_pmts.shape[0])
        if not self.use_orientations:
            data = np.zeros((5, self.n_points))
        else:
            hit_orientations = self.geo_orientations[hit_pmts[:n_hits], :]
            data = np.zeros((7, self.n_points))
            data[3:5, :n_hits] = hit_orientations.T
        data[:3, :n_hits] = hit_positions[:n_hits].T
        data[-2, :n_hits] = hit_charges[:n_hits]
        data[-1, :n_hits] = hit_times[:n_hits]

        data = du.apply_random_transformations(self.transforms, data)

        data_dict["data"] = data
        return data_dict