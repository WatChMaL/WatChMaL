"""
Class implementing a mPMT dataset for CNNs with segmentation
"""

# hydra imports
from hydra.utils import instantiate

# torch imports
from torch.utils.data import Dataset

# generic imports
import numpy as np
import pickle

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5TrueDataset
import watchmal.dataset.data_utils as du

class CNNmPMTSegmentationDataset(Dataset):
    def __init__(self, digi_dataset_config, true_hits_h5file, digi_truth_mapping_file, valid_parents=(-1, 2, 3),
                 transform_segmentation = True):
        """
        Args:
            digi_dataset_config     ... config for dataset for digitized hits
            true_hits_h5file        ... path to h5 dataset file for true hits
            digi_truth_mapping_file ... path to file with a pickled list mapping digitized hit events to true hit events
            valid_parents           ... valid ID values for hit parents

        """
        self.digi_dataset = instantiate(digi_dataset_config)
        if transform_segmentation:
            self.transforms = self.digi_dataset.transforms
            self.digi_dataset.transforms = None
        else:
            self.transforms = None
        self.truth_dataset = H5TrueDataset(true_hits_h5file, transforms=None, digitize_hits=False)
        with open(digi_truth_mapping_file, 'rb') as f:
            self.digi_truth_mapping = pickle.load(f)
        self.valid_parents = valid_parents

    def get_digi_hit_parent(self, digi_hit_pmt, true_hit_pmt, true_hit_parent):
        digi_hit_parent_count = {}
        for p in self.valid_parents:
            parent_true_hits = np.where(true_hit_parent == p)
            parent_hit_pmts = true_hit_pmt[parent_true_hits]
            digi_hit_parent_count[p] = np.isin(digi_hit_pmt, parent_hit_pmts)
        digi_hit_parent = np.zeros(digi_hit_pmt.shape)
        for p in self.valid_parents:
            is_this_parent = np.ones(digi_hit_pmt.shape)
            for o in self.valid_parents:
                if o == p:
                    is_this_parent &= digi_hit_parent_count[p]
                else:
                    is_this_parent &= ~digi_hit_parent_count[p]
            digi_hit_parent[is_this_parent] = p
        return digi_hit_parent

    def __getitem__(self, item):

        data_dict = self.digi_dataset.__getitem__(item)

        truth_item = self.digi_truth_mapping[item]
        self.truth_dataset.__getitem__(truth_item)
        parents = self.get_digi_hit_parent(self.digi_dataset.event_hit_pmts, self.truth_dataset.event_hit_pmts, self.truth_dataset.event_hit_parents)

        segmentation = self.digi_dataset.process_data(self.digi_dataset.event_hit_pmts, parents)

        if self.transforms is not None:
            data, segmentation = du.apply_random_transformations(self.transforms, data_dict["data"], segmentation)
            data_dict["data"] = data

        data_dict["segmentation"] = segmentation

        return data_dict