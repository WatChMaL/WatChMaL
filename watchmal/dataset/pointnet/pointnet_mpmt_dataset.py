import numpy as np
import torch

from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du

barrel_map_array_idxs = np.array([ 6,  7,  8,  9, 10, 11,  0,  1,  2,  3,  4,  5, 15, 16, 17, 12, 13, 14, 18], dtype=np.int16)
pmts_per_mpmt = 19


class PointNetMultiPMTDataset(H5Dataset):

    def __init__(self, h5file, geometry_file, is_distributed, use_orientations=False, transforms=None, max_points=None):
        super().__init__(h5file, is_distributed)
        geo_file = np.load(geometry_file, 'r')
        geo_positions = torch.from_numpy(geo_file["position"]).float()
        geo_orientations = torch.from_numpy(geo_file["orientation"]).float()
        self.mpmt_positions = geo_positions[18::19, :].T
        self.mpmt_orientations = geo_orientations[18::19, :].T
        mpmt_y = np.abs(self.mpmt_positions[1, :])
        self.barrel_mpmts = np.where(mpmt_y < mpmt_y.max() - 10)[0].astype(np.int16)
        self.use_orientations = use_orientations
        self.max_points = max_points
        self.transforms = du.get_transformations(transformations, transforms)


    def  __getitem__(self, item):

        data_dict = super().__getitem__(item)
        hit_pmts = data_dict["data"]["hit_pmts"]
        hit_charges = data_dict["data"]["hit_charges"]
        hit_times = data_dict["data"]["hit_times"]

        hit_mpmts = hit_pmts // pmts_per_mpmt
        hit_pmt_in_modules = hit_pmts % pmts_per_mpmt
        hit_barrel = np.where(np.in1d(hit_mpmts, self.barrel_mpmts))[0]
        hit_pmt_in_modules[hit_barrel] = barrel_map_array_idxs[hit_pmt_in_modules[hit_barrel]]

        if self.max_points is not None:
            n_points = self.max_points
            unique_mpmts, unique_hit_mpmts = np.unique(hit_mpmts, return_index=True)
            if unique_mpmts.shape[0] > self.max_points:
                unique_mpmts = unique_mpmts[:self.max_points]
                unique_hit_mpmts = unique_hit_mpmts.where[unique_hit_mpmts < self.max_points]
        else:
            n_points = self.mpmt_positions.shape[1]
            unique_mpmts = np.arange(n_points)
            unique_hit_mpmts = hit_mpmts
        if not self.use_orientations:
            data = np.zeros((41, n_points))
            charge_channels = hit_pmt_in_modules+3
            time_channels = hit_pmt_in_modules+22
        else:
            data = np.zeros((44, n_points))
            data[3:5,:] = self.mpmt_orientations[:, unique_mpmts]
            charge_channels = hit_pmt_in_modules+6
            time_channels = hit_pmt_in_modules+25
        data[:3, :] = self.mpmt_positions[:, unique_mpmts]
        data[charge_channels, unique_hit_mpmts] = hit_charges
        data[time_channels, unique_hit_mpmts] = hit_times

        data = du.apply_random_transformations(self.transforms, data)

        data_dict["data"] = data
        return data_dict