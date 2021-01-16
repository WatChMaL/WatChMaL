import numpy as np
import torch

from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations

barrel_map_array_idxs = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18]
pmts_per_mpmt = 19


class PointNetMultiPMTDataset(H5Dataset):

    def __init__(self, h5file, geometry_file, is_distributed, use_orientations=False, transforms=None, max_points=None):
        super().__init__(h5file, is_distributed, transforms)
        geo_file = np.load(geometry_file, 'r')
        self.geo_positions = torch.from_numpy(geo_file["position"]).float()
        self.geo_orientations = torch.from_numpy(geo_file["orientation"]).float()
        self.use_orientations = use_orientations
        self.transforms = transforms
        self.max_points = max_points
        if self.transforms is not None:
            for transform_name in transforms:
                assert hasattr(transformations, transform_name), f"Error: There is no defined transform named {transform_name}"
            transform_funcs = [getattr(transformations, transform_name) for transform_name in transforms]
            self.transforms = transform_funcs
            self.n_transforms = len(self.transforms)


    def get_data(self, hit_pmts, hit_charges, hit_times):
        hit_mpmts = hit_pmts // pmts_per_mpmt
        hit_pmt_in_modules = hit_pmts % pmts_per_mpmt
        if self.max_points is not None:
            n_points = self.max_points
            unique_mpmts, unique_hit_mpmts = np.unique(hit_mpmts, return_index=True)
            if unique_mpmts.shape[0] > self.max_points:
                unique_mpmts = unique_mpmts[:self.max_points]
                unique_hit_mpmts = unique_hit_mpmts.where[unique_hit_mpmts < self.max_points]
        else:
            n_points = self.geo_positions.shape[0] / pmts_per_mpmt
            unique_mpmts = np.arange(n_points)
            unique_hit_mpmts = hit_mpmts
        if not self.use_orientations:
            data = np.zeros((41, n_points))
            charge_channels = hit_pmt_in_modules+3
            time_channels = hit_pmt_in_modules+22
        else:
            data = np.zeros((44, n_points))
            mpmt_orientations = self.geo_orientations[18::19, :].T
            data[3:5,:] = mpmt_orientations[unique_mpmts,:]
            charge_channels = hit_pmt_in_modules+6
            time_channels = hit_pmt_in_modules+25
        mpmt_positions = self.geo_positions[18::19, :].T
        data[:3, :] = mpmt_positions[unique_mpmts,:]
        data[charge_channels, unique_hit_mpmts] = hit_charges
        data[time_channels, unique_hit_mpmts] = hit_times

        if self.transforms is not None:
            selection = np.random.choice(2, self.n_transforms)
            for i, transform in enumerate(self.transforms):
                if selection[i]:
                    data = transform(data)

        return data