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
                 parent_type="max", transform_segmented_labels = True):
        """
        Args:
            digi_dataset_config     ... config for dataset for digitized hits
            true_hits_h5file        ... path to h5 dataset file for true hits
            digi_truth_mapping_file ... path to file with a pickled list mapping digitized hit events to true hit events
            valid_parents           ... valid ID values for hit parents

        """
        self.digi_dataset = instantiate(digi_dataset_config)
        if transform_segmented_labels:
            self.transforms = self.digi_dataset.transforms
            self.digi_dataset.transforms = None
        else:
            self.transforms = None
        self.truth_dataset = H5TrueDataset(true_hits_h5file, digitize_hits=False)
        with open(digi_truth_mapping_file, 'rb') as f:
            self.digi_truth_mapping = pickle.load(f)
        self.valid_parents = np.array(valid_parents)
        if parent_type=="only":
            self.get_digi_hit_parent = self.get_digi_hit_only_parent
        elif parent_type=="max":
            self.get_digi_hit_parent = self.get_digi_hit_max_parent

    def get_digi_hit_only_parent(self, digi_hit_pmt, true_hit_pmt, true_hit_parent):
        # find which digi hits have which parents
        digi_hit_has_parent = np.zeros((len(self.valid_parents), len(digi_hit_pmt)))
        for i, p in enumerate(self.valid_parents):
            parent_true_hits = np.where(true_hit_parent == p)
            parent_hit_pmts = true_hit_pmt[parent_true_hits]
            digi_hit_has_parent[i] = np.isin(digi_hit_pmt, parent_hit_pmts)
        # choose the hit parent if it only has one parent, or -2 if it has multiple
        # first take the argmax (will be the only parent if there's only one parent, or first parent if there's multiple parents)
        digi_hit_parent = self.valid_parents[np.argmax(digi_hit_has_parent, axis=0)]
        # replace with -2 for any with more than one parent
        digi_hit_parent[np.where(np.count_nonzero(digi_hit_has_parent, axis=0)>1)] = -2
        return digi_hit_parent

    def get_digi_hit_max_parent(self, digi_hit_pmt, true_hit_pmt, true_hit_parent):
        # sort the digi hit pmts for fast processing
        sort = np.argsort(digi_hit_pmt)
        unsort = np.empty(sort.shape, dtype=int)
        unsort[sort] = np.arange(len(sort))
        digi_hit_pmt_sorted = digi_hit_pmt[sort]
        # find the (sorted) index of the digi hit for each true hit, and mask those that have no digi hit
        true_hit_sorted_digi_index = np.searchsorted(digi_hit_pmt_sorted, true_hit_pmt)
        true_hit_sorted_digi_index[true_hit_sorted_digi_index == len(digi_hit_pmt)] = 0
        mask = digi_hit_pmt_sorted[true_hit_sorted_digi_index] == true_hit_pmt
        # count the number of true hits of given parent type for each digihit
        digi_hit_parent_count = np.zeros((len(self.valid_parents), len(digi_hit_pmt)))
        for i, p in enumerate(self.valid_parents):
            parent_hits_sorted_digi_index = true_hit_sorted_digi_index[mask & (true_hit_parent == p)]
            sorted_counts = np.zeros(digi_hit_pmt.shape, dtype=int)
            np.add.at(sorted_counts, parent_hits_sorted_digi_index, 1)
            digi_hit_parent_count[i] = sorted_counts[unsort]
        # choose the hit parent based on which parent has the highest count
        count_argmax = np.argmax(digi_hit_parent_count, axis=0)
        digi_hit_parent = self.valid_parents[count_argmax]
        # replace with -2 for any digi hits with equal max count of more than one parent
        count_max = digi_hit_parent_count[count_argmax, np.arange(len(count_argmax))]
        digi_hit_parent[np.where(np.count_nonzero(digi_hit_parent_count==count_max, axis=0)>1)] = -2
        return digi_hit_parent

    def __getitem__(self, item):

        data_dict = self.digi_dataset.__getitem__(item)

        truth_item = self.digi_truth_mapping[item]
        self.truth_dataset.__getitem__(truth_item)
        parents = self.get_digi_hit_parent(self.digi_dataset.event_hit_pmts, self.truth_dataset.event_hit_pmts, self.truth_dataset.event_hit_parents)

        segmented_labels = self.digi_dataset.process_data(self.digi_dataset.event_hit_pmts, parents)

        if self.transforms is not None:
            data, segmented_labels = du.apply_random_transformations(self.transforms, data_dict["data"], segmented_labels)
            data_dict["data"] = data

        data_dict["segmented_labels"] = segmented_labels

        return data_dict
