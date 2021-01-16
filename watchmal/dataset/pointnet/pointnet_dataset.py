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

class PointNetDataset(H5Dataset):

    def __init__(self, h5file, geometry_file, is_distributed, use_orientations=False, n_points=4000, transforms=None):
        super().__init__(h5file, is_distributed, transforms)
        geo_file = np.load(geometry_file, 'r')
        self.geo_positions = torch.from_numpy(geo_file["position"]).float()
        self.geo_orientations = torch.from_numpy(geo_file["orientation"]).float()
        self.use_orientations = use_orientations
        self.n_points = n_points
        self.transforms = transforms
        if self.transforms is not None:
            for transform_name in transforms:
                assert hasattr(transformations, transform_name), f"Error: There is no defined transform named {transform_name}"
            transform_funcs = [getattr(transformations, transform_name) for transform_name in transforms]
            self.transforms = transform_funcs
            self.n_transforms = len(self.transforms)


    def get_data(self, hit_pmts, hit_charges, hit_times):
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

        if self.transforms is not None:
            selection = np.random.choice(2, self.n_transforms)
            for i, transform in enumerate(self.transforms):
                if selection[i]:
                    data = transform(data)

        return data
