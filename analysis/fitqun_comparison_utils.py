"""
Utils for comparing performance to fiTQun
"""

import numpy as np
import pickle
import uproot
import h5py
import matplotlib.pyplot as plt
import re
from functools import reduce
from tqdm import tqdm_notebook as tqdm


def load_mu_fq_output(fq_mapping_path, gamma_file_path, e_file_path, mu_file_path, pion_file_path):
    return load_fq_output(fq_mapping_path, gamma_file_path, e_file_path, mu_file_path, pion_file_path, comparison='e_v_mu')


def load_gamma_fq_output(fq_mapping_path, gamma_file_path, e_file_path, mu_file_path, pion_file_path, discriminator='e_v_gamma'):
    return load_fq_output(fq_mapping_path, gamma_file_path, e_file_path, mu_file_path, pion_file_path, comparison='e_v_gamma', discriminator=discriminator)


def load_pion_fq_output(fq_mapping_path, gamma_file_path, e_file_path, mu_file_path, pion_file_path):
    return load_fq_output(fq_mapping_path, gamma_file_path, e_file_path, mu_file_path, pion_file_path, comparison='e_v_pion')


def load_fq_output(fq_mapping_path, gamma_file_path, e_file_path, mu_file_path, pion_file_path, comparison, discriminator=None):
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

    with open(fq_mapping_path, 'rb') as handle:
        fq_mapping = pickle.load(handle)

    gamma_fq_indices = fq_mapping['gamma_fq_indices']
    e_fq_indices     = fq_mapping['e_fq_indices']
    mu_fq_indices    = fq_mapping['mu_fq_indices']
    
    pion_fq_indices  = fq_mapping['pion_fq_indices']
    

    # Load fiTQun results
    gamma_file_data = uproot.open(gamma_file_path)['fiTQun;1']
    e_file_data     = uproot.open(e_file_path)['fiTQun;1']
    mu_file_data    = uproot.open(mu_file_path)['fiTQun;1']
    
    pion_file_data  = uproot.open(pion_file_path)['fiTQun;1']
    

    # Load gamma results
    gamma_set_nll = gamma_file_data.arrays('fq1rnll')['fq1rnll']

    gamma_set_e_nll, gamma_set_mu_nll = gamma_set_nll[:, 0, 1], gamma_set_nll[:, 0, 2]

    gamma_set_gamma_nll = gamma_file_data.arrays('fq2elecnll')['fq2elecnll']

    gamma_fqpi0mom1 = gamma_file_data.arrays('fqpi0mom1')['fqpi0mom1'][:, 0]
    gamma_fqpi0mom2 = gamma_file_data.arrays('fqpi0mom2')['fqpi0mom2'][:, 0]
    gamma_fqpi0nll  = gamma_file_data.arrays('fqpi0nll')['fqpi0nll'][:, 0]
    gamma_fqpi0mass = gamma_file_data.arrays('fqpi0mass')['fqpi0mass'][:, 0]

    # Load electron results
    e_set_nll    = e_file_data.arrays('fq1rnll')['fq1rnll']

    e_set_e_nll, e_set_mu_nll = e_set_nll[:, 0, 1], e_set_nll[:, 0, 2]

    e_set_gamma_nll = e_file_data.arrays('fq2elecnll')['fq2elecnll']

    e_fqpi0mom1 = e_file_data.arrays('fqpi0mom1')['fqpi0mom1'][:, 0]
    e_fqpi0mom2 = e_file_data.arrays('fqpi0mom2')['fqpi0mom2'][:, 0]
    e_fqpi0nll  = e_file_data.arrays('fqpi0nll')['fqpi0nll'][:, 0]
    e_fqpi0mass = e_file_data.arrays('fqpi0mass')['fqpi0mass'][:, 0]

    # Load mu results
    mu_set_nll   = mu_file_data.arrays('fq1rnll')['fq1rnll']

    mu_set_e_nll, mu_set_mu_nll = mu_set_nll[:, 0, 1], mu_set_nll[:, 0, 2]

    #mu_set_gamma_nll = mu_file_data.arrays('fq2elecnll')['fq2elecnll']

    mu_fqpi0mom1 = mu_file_data.arrays('fqpi0mom1')['fqpi0mom1'][:, 0]
    mu_fqpi0mom2 = mu_file_data.arrays('fqpi0mom2')['fqpi0mom2'][:, 0]
    mu_fqpi0nll  = mu_file_data.arrays('fqpi0nll')['fqpi0nll'][:, 0]
    mu_fqpi0mass = mu_file_data.arrays('fqpi0mass')['fqpi0mass'][:, 0]

    # Load pion results
    
    pion_set_nll   = pion_file_data.arrays('fq1rnll')['fq1rnll']

    pion_set_e_nll, pion_set_mu_nll = pion_set_nll[:, 0, 1], pion_set_nll[:, 0, 2]
    
    #pion_set_gamma_nll = pion_file_data.arrays('fq2elecnll')['fq2elecnll']

    pion_fqpi0mom1 = pion_file_data.arrays('fqpi0mom1')['fqpi0mom1'][:, 0]
    pion_fqpi0mom2 = pion_file_data.arrays('fqpi0mom2')['fqpi0mom2'][:, 0]
    pion_fqpi0nll  = pion_file_data.arrays('fqpi0nll')['fqpi0nll'][:, 0]
    pion_fqpi0mass = pion_file_data.arrays('fqpi0mass')['fqpi0mass'][:, 0]
    

    # Define discriminators
    # (false_label_nll - true_label_nll)
    if comparison == 'e_v_mu':
        e_set_discriminator     = np.array(e_set_mu_nll - e_set_e_nll)
        mu_set_discriminator    = np.array(mu_set_mu_nll - mu_set_e_nll) 
        gamma_set_discriminator = np.array(gamma_set_mu_nll - gamma_set_e_nll)
        
        pion_set_discriminator  = np.array(pion_set_mu_nll - pion_set_e_nll)
        
    elif comparison == 'e_v_gamma':
        # NOTE: mu outputs currently don't have 2elec fit
        if discriminator == 'e_v_gamma':
            gamma_set_discriminator = np.array(gamma_set_gamma_nll - gamma_set_e_nll)
            e_set_discriminator     = np.array(e_set_gamma_nll - e_set_e_nll)
            mu_set_discriminator    = np.array( - mu_set_e_nll) 
            
            pion_set_discriminator  = np.array( - pion_set_e_nll)
            
        elif discriminator == 'e_v_mu':
            e_set_discriminator     = np.array(e_set_e_nll - e_set_mu_nll)
            mu_set_discriminator    = np.array(mu_set_e_nll - mu_set_mu_nll) 
            gamma_set_discriminator = np.array(gamma_set_e_nll - gamma_set_mu_nll)
            
            pion_set_discriminator  = np.array(pion_set_e_nll - pion_set_mu_nll)
            
        elif discriminator == 'gamma_v_mu':
            gamma_set_discriminator = np.array(gamma_set_gamma_nll - gamma_set_mu_nll)
            e_set_discriminator     = np.array(e_set_gamma_nll - e_set_mu_nll)
            mu_set_discriminator    = np.array( - mu_set_e_nll) 
            
            pion_set_discriminator  = np.array( - pion_set_mu_nll)
            
    elif comparison == 'e_v_pion':
        gamma_set_discriminator = np.array(gamma_set_e_nll - gamma_fqpi0nll)
        e_set_discriminator     = np.array(e_fqpi0nll - e_set_e_nll)
        mu_set_discriminator    = np.array(mu_fqpi0nll - mu_set_e_nll)
        pion_set_discriminator  = np.array(pion_fqpi0nll - pion_set_e_nll)

    # Construct likelihoods
    fq_likelihoods = np.concatenate((e_set_discriminator[e_fq_indices],
                                     mu_set_discriminator[mu_fq_indices],
                                     gamma_set_discriminator[gamma_fq_indices]
                                     ,
                                     pion_set_discriminator[pion_fq_indices]
                                     
                                     ))

    # Collect scores
    fq_scores = np.zeros((fq_likelihoods.shape[0], 3))
    fq_scores[:, 1] = fq_likelihoods

    # Generate labels
    fq_labels = np.concatenate((np.ones_like(e_set_discriminator[e_fq_indices])*1,
                                np.ones_like(mu_set_discriminator[mu_fq_indices])*2,
                                np.ones_like(gamma_set_discriminator[gamma_fq_indices])*0
                                ,
                                np.ones_like(pion_set_discriminator[pion_fq_indices])*3
                                ))
    
    # Collect reconstructed momentum values
    if comparison == 'e_v_pion':
        gamma_set_mom = np.array(gamma_fqpi0mom1 + gamma_fqpi0mom2)
        e_set_mom     = np.array(e_fqpi0mom1 + e_fqpi0mom2)
        mu_set_mom    = np.array(mu_fqpi0mom1 + mu_fqpi0mom2)
        pion_set_mom  = np.array(pion_fqpi0mom1 + pion_fqpi0mom2)
    else:
        gamma_set_mom = np.array(gamma_file_data.arrays('fq1rmom')['fq1rmom'][:, 0, 1])
        e_set_mom     = np.array(e_file_data.arrays('fq1rmom')['fq1rmom'][:, 0, 1])
        mu_set_mom    = np.array(mu_file_data.arrays('fq1rmom')['fq1rmom'][:, 0, 1])
        
        pion_set_mom  = np.array(pion_file_data.arrays('fq1rmom')['fq1rmom'][:, 0, 1])
        

    fq_mom = np.concatenate((e_set_mom[e_fq_indices],
                             mu_set_mom[mu_fq_indices],
                             gamma_set_mom[gamma_fq_indices]
                             ,
                             pion_set_mom[pion_fq_indices]
                             ))
    
    # Collect reconstructed mass values
    gamma_set_mass = np.array(gamma_fqpi0mass)
    e_set_mass     = np.array(e_fqpi0mass)
    mu_set_mass    = np.array(mu_fqpi0mass)
    
    pion_set_mass  = np.array(pion_fqpi0mass)
    

    fq_masses = np.concatenate((e_set_mass[e_fq_indices],
                                mu_set_mass[mu_fq_indices],
                                gamma_set_mass[gamma_fq_indices]
                                ,
                                pion_set_mass[pion_fq_indices]
                                ))

    return fq_scores, fq_labels, fq_mom, fq_masses


def deprecated_load_fq_output(mapping_indices_path, fq_failed_idxs_path, test_idxs_path, cut_path, cut_list):
    '''
    load_fq_output(mapping_indices_path, test_idxs, cut_path, cut_list)
    
    Purpose : Load FiTQun output in old (full tank) framework format
    
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

def deprecated_verify_fq_matching(fq_rootfiles, fq_eventids, filtered_dataset_rootfiles, filtered_dataset_eventids):
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