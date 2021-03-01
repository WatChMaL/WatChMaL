"""
Utils for comparing performance to fiTQun
"""

import numpy as np
import h5py
import re
from functools import reduce
from tqdm import tqdm_notebook as tqdm

def load_fq_output(mapping_indices_path, fq_failed_idxs_path, test_idxs_path, cut_path, cut_list):
    # TODO: rework to fit current framework
    '''
    load_fq_output(mapping_indices_path, test_idxs, cut_path, cut_list)
    
    Purpose : Load FiTQun output matching the desired 
    
    Args: mapping_indices_path ... path to .npz list of mapping indices for FiTQun
          fq_failed_idxs_path  ... path to .npz containing indices of events (in test set ordering) for which FiTQun failed to produce output 
          test_idxs_path       ... path to .npz containing indices in pointnet set - idx array must be titled 'test_idxs' in the archive
          cut_path             ... path to pointnet cuts npz file
          cut_list             ... list of cuts to be applied. Must be an array in the .npz pointed to be cut_path
    author: Calum Macdonald   
    August 2020
    '''
    ###### Load the fiTQun results ######

    # File paths for fiTQun results
    fiTQun_gamma_path = "/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_gamma.npz"
    fiTQun_e_path     = "/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_e-.npz"
    fiTQun_mu_path    = "/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_mu-.npz"

    # Load fiTQun results
    f_gamma = np.load(fiTQun_gamma_path, allow_pickle=True)
    f_e     = np.load(fiTQun_e_path, allow_pickle=True)
    f_mu    = np.load(fiTQun_mu_path, allow_pickle=True)

    fq_filename_original = (f_gamma['filename'], f_e['filename'], f_mu['filename'])
    fq_eventids_original = (f_gamma['eventid'], f_e['eventid'], f_mu['eventid'])
    fq_flag_original     = (f_gamma['flag'], f_e['flag'], f_mu['flag'])
    fq_nll_original      = (f_gamma['nLL'], f_e['nLL'], f_mu['nLL'])
    fq_mom_original      = (f_gamma['momentum'],f_e['momentum'],f_mu['momentum'])

    n_events = int(reduce(lambda x,y : x+y, list(map(lambda x : x.shape[0], fq_filename_original))))

    fq_rootfiles = np.empty(n_events,dtype=object)
    fq_eventids  = np.zeros(n_events)
    fq_flag      = np.empty((n_events, 2))
    fq_nll       = np.empty((n_events, 2))
    fq_mom       = np.empty((n_events, 2))

    fq_mapping_indices = np.load(mapping_indices_path,allow_pickle=True)['arr_0']

    fq_failed_idxs = np.load(fq_failed_idxs_path, allow_pickle = True)['failed_indices_pointing_to_h5_test_set'].astype(int)
    filtered_indices = np.load("/fast_scratch/WatChMaL/data/IWCD_fulltank_300_pe_idxs.npz", allow_pickle=True)
    test_filtered_indices = filtered_indices['test_idxs']
    stest_filtered_indices = np.delete(test_filtered_indices, fq_failed_idxs,0)

    idx_dic = {}
    for i, idx in enumerate(stest_filtered_indices):
        idx_dic[idx] = i

    test_idxs = np.load(test_idxs_path, allow_pickle=True)['test_idxs']

    keep_idxs = []
    stest_idxs = []
    i=0
    for idx in test_idxs:
        try:
            keep_idxs.append(idx_dic[idx])
            stest_idxs.append(idx)
        except KeyError:
            i+=1

    f = h5py.File("/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_full_tank_pointnet.h5", "r")
    original_labels = np.array(f['labels'])
    labels = np.array(original_labels[test_filtered_indices])
    f.close()
    slabels = np.delete(labels, fq_failed_idxs, 0)

    for i,ptype in enumerate(slabels):
        fq_rootfiles[i] = str(fq_filename_original[ptype][fq_mapping_indices[i]])
        fq_eventids[i] = fq_eventids_original[ptype][fq_mapping_indices[i]]
        fq_flag[i] = fq_flag_original[ptype][fq_mapping_indices[i]]
        fq_nll[i] = fq_nll_original[ptype][fq_mapping_indices[i]]
        fq_mom[i] = fq_mom_original[ptype][fq_mapping_indices[i]]
    
    fq_scores = np.zeros((fq_nll.shape[0],3))
    fq_scores[:,0] = fq_nll[:,1] - fq_nll[:,0]
    fq_scores[:,1] = fq_nll[:,1] - fq_nll[:,0]
    fq_scores[:,2] = fq_nll[:,0] - fq_nll[:,1]
    fq_labels = slabels

    fq_rootfiles = fq_rootfiles[keep_idxs]
    fq_eventids = fq_eventids[keep_idxs]
    fq_flag = fq_flag[keep_idxs]
    fq_scores = fq_scores[keep_idxs]
    fq_mom = fq_mom[keep_idxs]
    fq_labels = fq_labels[keep_idxs]

    fq_rootfiles = apply_cuts(fq_rootfiles, stest_idxs, cut_path, cut_list)
    fq_eventids = apply_cuts(fq_eventids, stest_idxs, cut_path, cut_list)
    fq_flag = apply_cuts(fq_flag, stest_idxs, cut_path, cut_list)
    fq_scores = apply_cuts(fq_scores, stest_idxs, cut_path, cut_list)
    fq_mom = apply_cuts(fq_mom, stest_idxs, cut_path, cut_list)
    fq_labels = apply_cuts(fq_labels, stest_idxs, cut_path, cut_list)

    fq_outputs = {'fq_rootfiles':fq_rootfiles, 
                  'fq_eventids':fq_eventids, 
                  'fq_flag':fq_flag, 
                  'fq_scores':fq_scores, 
                  'fq_mom':fq_mom, 
                  'fq_labels':fq_labels}

    return fq_outputs

def deprecated_load_fq_output(mapping_indices_path, fq_failed_idxs_path, test_idxs_path, cut_path, cut_list):
    # TODO: rework to fit current framework
    '''
    load_fq_output(mapping_indices_path, test_idxs, cut_path, cut_list)
    
    Purpose : Load FiTQun output matching the desired 
    
    Args: mapping_indices_path ... path to .npz list of mapping indices for FiTQun
          fq_failed_idxs_path  ... path to .npz containing indices of events (in test set ordering) for which FiTQun failed to produce output 
          test_idxs_path       ... path to .npz containing indices in pointnet set - idx array must be titled 'test_idxs' in the archive
          cut_path             ... path to pointnet cuts npz file
          cut_list             ... list of cuts to be applied. Must be an array in the .npz pointed to be cut_path
    author: Calum Macdonald   
    August 2020
    '''
    ###### Load the fiTQun results ######

    # File paths for fiTQun results
    fiTQun_gamma_path = "/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_gamma.npz"
    fiTQun_e_path     = "/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_e-.npz"
    fiTQun_mu_path    = "/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_mu-.npz"

    # Load fiTQun results
    f_gamma = np.load(fiTQun_gamma_path, allow_pickle=True)
    f_e     = np.load(fiTQun_e_path, allow_pickle=True)
    f_mu    = np.load(fiTQun_mu_path, allow_pickle=True)

    fq_filename_original = (f_gamma['filename'], f_e['filename'], f_mu['filename'])
    fq_eventids_original = (f_gamma['eventid'], f_e['eventid'], f_mu['eventid'])
    fq_flag_original     = (f_gamma['flag'], f_e['flag'], f_mu['flag'])
    fq_nll_original      = (f_gamma['nLL'], f_e['nLL'], f_mu['nLL'])
    fq_mom_original      = (f_gamma['momentum'],f_e['momentum'],f_mu['momentum'])

    n_events = int(reduce(lambda x,y : x+y, list(map(lambda x : x.shape[0], fq_filename_original))))

    fq_rootfiles = np.empty(n_events,dtype=object)
    fq_eventids  = np.zeros(n_events)
    fq_flag      = np.empty((n_events, 2))
    fq_nll       = np.empty((n_events, 2))
    fq_mom       = np.empty((n_events, 2))

    fq_mapping_indices = np.load(mapping_indices_path,allow_pickle=True)['arr_0']

    fq_failed_idxs = np.load(fq_failed_idxs_path, allow_pickle = True)['failed_indices_pointing_to_h5_test_set'].astype(int)
    filtered_indices = np.load("/fast_scratch/WatChMaL/data/IWCD_fulltank_300_pe_idxs.npz", allow_pickle=True)
    test_filtered_indices = filtered_indices['test_idxs']
    stest_filtered_indices = np.delete(test_filtered_indices, fq_failed_idxs,0)

    idx_dic = {}
    for i, idx in enumerate(stest_filtered_indices):
        idx_dic[idx] = i

    test_idxs = np.load(test_idxs_path, allow_pickle=True)['test_idxs']

    keep_idxs = []
    stest_idxs = []
    i=0
    for idx in test_idxs:
        try:
            keep_idxs.append(idx_dic[idx])
            stest_idxs.append(idx)
        except KeyError:
            i+=1

    f = h5py.File("/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_full_tank_pointnet.h5", "r")
    original_labels = np.array(f['labels'])
    labels = np.array(original_labels[test_filtered_indices])
    f.close()
    slabels = np.delete(labels, fq_failed_idxs, 0)

    for i,ptype in enumerate(slabels):
        fq_rootfiles[i] = str(fq_filename_original[ptype][fq_mapping_indices[i]])
        fq_eventids[i] = fq_eventids_original[ptype][fq_mapping_indices[i]]
        fq_flag[i] = fq_flag_original[ptype][fq_mapping_indices[i]]
        fq_nll[i] = fq_nll_original[ptype][fq_mapping_indices[i]]
        fq_mom[i] = fq_mom_original[ptype][fq_mapping_indices[i]]
    
    fq_scores = np.zeros((fq_nll.shape[0],3))
    fq_scores[:,0] = fq_nll[:,1] - fq_nll[:,0]
    fq_scores[:,1] = fq_nll[:,1] - fq_nll[:,0]
    fq_scores[:,2] = fq_nll[:,0] - fq_nll[:,1]
    fq_labels = slabels

    fq_rootfiles = fq_rootfiles[keep_idxs]
    fq_eventids = fq_eventids[keep_idxs]
    fq_flag = fq_flag[keep_idxs]
    fq_scores = fq_scores[keep_idxs]
    fq_mom = fq_mom[keep_idxs]
    fq_labels = fq_labels[keep_idxs]

    fq_rootfiles = apply_cuts(fq_rootfiles, stest_idxs, cut_path, cut_list)
    fq_eventids = apply_cuts(fq_eventids, stest_idxs, cut_path, cut_list)
    fq_flag = apply_cuts(fq_flag, stest_idxs, cut_path, cut_list)
    fq_scores = apply_cuts(fq_scores, stest_idxs, cut_path, cut_list)
    fq_mom = apply_cuts(fq_mom, stest_idxs, cut_path, cut_list)
    fq_labels = apply_cuts(fq_labels, stest_idxs, cut_path, cut_list)

    fq_outputs = {'fq_rootfiles':fq_rootfiles, 
                  'fq_eventids':fq_eventids, 
                  'fq_flag':fq_flag, 
                  'fq_scores':fq_scores, 
                  'fq_mom':fq_mom, 
                  'fq_labels':fq_labels}

    return fq_outputs

def verify_fq_matching(fq_rootfiles, fq_eventids, filtered_dataset_rootfiles, filtered_dataset_eventids):
    description = 'Verification Progress: '

    for i in tqdm(range(len(fq_rootfiles)), desc=description, position=pos):
        # verify matching rootfiles
        assert re.sub('_fiTQun','',fq_rootfiles[i].split('/')[-1]) == preferred_run['rootfiles'][i].split('/')[-1]

        # verify matcheing eventids
        assert fq_eventids[i]-1 == preferred_run['eventids'][i]

    assert len(preferred_run['rootfiles']) == fq_rootfiles.shape[0]
    print("Success! We now have a FiTQun output set in the same order as the h5 test set")


def apply_cuts(array, idxs, cut_path, cut_list):
    # TODO: replace with updated cutting function
    '''
    apply_cuts(array, indices, cut_path, cut_list)

    Purpose: Applies cuts to a given array, based on the given cut file and indices.

    Args:   array               ... 1d array of length n, that we wish to cut 
            idxs                ... 1d array of length n, where the ith entry gives the index of the event in the
                                    pointnet test set ordering corresponding to the ith entry in array
            cut_path            ... path to pointnet cuts npz file
            cut_list            ... list of cuts to be applied. Must be an array in the .npz pointed to be cut_path
    '''
    cut_file = np.load(cut_path, allow_pickle=True) 

    cut_arrays = []
    for cut in cut_list:
        assert cut in cut_file.keys(), f"Error, {cut} has no associated cut file"
        cut_arrays.append(cut_file[cut][idxs])

    combined_cut_array=np.array(list(map(lambda x : 1 if 1 in x else 0,  list(zip(*cut_arrays)))))
    cut_idxs = np.where(combined_cut_array==1)[0]    

    return np.delete(array, cut_idxs, 0)