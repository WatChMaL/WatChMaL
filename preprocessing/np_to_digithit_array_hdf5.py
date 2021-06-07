import numpy as np
import os
import sys
import subprocess
from datetime import datetime
import argparse
import h5py
import pos_utils as pu

def get_args():
    parser = argparse.ArgumentParser(description='convert and merge .npz files to hdf5')
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('-o', '--output_file', type=str)
    parser.add_argument('-H', '--half-height', type=float, default=300)
    parser.add_argument('-R', '--radius', type=float, default=400)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = get_args()
    print("ouput file:", config.output_file)
    f = h5py.File(config.output_file, 'w')
    
    script_path = os.path.dirname(os.path.abspath(__file__))
    git_status = subprocess.check_output(['git', '-C', script_path, 'status', '--porcelain', '--untracked-files=no']).decode()
    '''
    if git_status:
        raise Exception("Directory of this script ({}) is not a clean git directory:\n{}Need a clean git directory for storing script version in output file.".format(script_path, git_status))
    git_describe = subprocess.check_output(['git', '-C', script_path, 'describe', '--always', '--long', '--tags']).decode().strip()
    print("git describe for path to this script ({}):".format(script_path), git_describe)
    f.attrs['git-describe'] = git_describe
    f.attrs['command'] = str(sys.argv)
    f.attrs['timestamp'] = str(datetime.now())
'''
    total_rows = 0
    total_hits = 0
    min_hits = 1
    good_rows = 0
    good_hits = 0
    print("counting events and hits, in files")
    file_event_triggers = {}
    for input_file in config.input_files:
        print(input_file, flush=True)
        if not os.path.isfile(input_file):
            raise ValueError(input_file+" does not exist")
        npz_file = np.load(input_file, allow_pickle=True)
        trigger_times = npz_file['trigger_time']
        trigger_types = npz_file['trigger_type']
        hit_triggers = npz_file['digi_hit_trigger']
        total_rows += hit_triggers.shape[0]
        event_triggers = np.full(hit_triggers.shape[0], np.nan)
        for i, (times, types, hit_trigs) in enumerate(zip(trigger_times, trigger_types, hit_triggers)):
            good_triggers = np.where(types==0)[0]
            if len(good_triggers)==0:
                continue
            first_trigger = good_triggers[np.argmin(times[good_triggers])]
            nhits = np.count_nonzero(hit_trigs==first_trigger)
            total_hits += nhits
            if nhits >= min_hits:
                event_triggers[i] = first_trigger
                good_hits += nhits
                good_rows += 1
        file_event_triggers[input_file] = event_triggers
    
    print(len(config.input_files), "files with", total_rows, "events with ", total_hits, "hits")
    print(good_rows, "events with at least", min_hits, "hits for a total of", good_hits, "hits")

    dset_labels=f.create_dataset("labels",
                                 shape=(good_rows,),
                                 dtype=np.int32)
    dset_PATHS=f.create_dataset("root_files",
                                shape=(good_rows,),
                                dtype=h5py.special_dtype(vlen=str))
    dset_IDX=f.create_dataset("event_ids",
                              shape=(good_rows,),
                              dtype=np.int32)
    dset_hit_time=f.create_dataset("hit_time",
                                 shape=(good_hits, ),
                                 dtype=np.float32)
    dset_hit_charge=f.create_dataset("hit_charge",
                                 shape=(good_hits, ),
                                 dtype=np.float32)
    dset_hit_pmt=f.create_dataset("hit_pmt",
                                  shape=(good_hits, ),
                                  dtype=np.int32)
    dset_event_hit_index=f.create_dataset("event_hits_index",
                                          shape=(good_rows,),
                                          dtype=np.int64) # int32 is too small to fit large indices
    dset_energies=f.create_dataset("energies",
                                   shape=(good_rows, 1),
                                   dtype=np.float32)
    dset_positions=f.create_dataset("positions",
                                    shape=(good_rows, 1, 3),
                                    dtype=np.float32)
    dset_angles=f.create_dataset("angles",
                                 shape=(good_rows, 2),
                                 dtype=np.float32)
    dset_veto = f.create_dataset("veto",
                                 shape=(good_rows,),
                                 dtype=np.bool_)
    dset_veto2 = f.create_dataset("veto2",
                                  shape=(good_rows,),
                                  dtype=np.bool_)

    offset = 0
    offset_next = 0
    hit_offset = 0
    hit_offset_next = 0
    #label_map = {22: 0, 11: 1, 13: 2}
    label_map = {0:0, 11:1}
    for input_file in config.input_files:
        print(input_file, flush=True)
        npz_file = np.load(input_file, allow_pickle=True)
        good_events = ~np.isnan(file_event_triggers[input_file])
        event_triggers = file_event_triggers[input_file][good_events]
        event_ids = npz_file['event_id'][good_events]
        root_files = npz_file['root_file'][good_events]
        pids = npz_file['pid'][good_events]
        positions = npz_file['position'][good_events]
        directions = npz_file['direction'][good_events]
        energies = npz_file['energy'][good_events]
        hit_times = npz_file['digi_hit_time'][good_events]
        hit_charges = npz_file['digi_hit_charge'][good_events]
        hit_pmts = npz_file['digi_hit_pmt'][good_events]
        hit_triggers = npz_file['digi_hit_trigger'][good_events]
        track_pid = npz_file['track_pid'][good_events]
        track_energy = npz_file['track_energy'][good_events]
        track_stop_position = npz_file['track_stop_position'][good_events]
        track_start_position = npz_file['track_start_position'][good_events]


        offset_next += event_ids.shape[0]

        dset_IDX[offset:offset_next] = event_ids
        dset_PATHS[offset:offset_next] = root_files
        dset_energies[offset:offset_next,:] = energies.reshape(-1,1)
        dset_positions[offset:offset_next,:,:] = positions.reshape(-1,1,3)

        labels = np.full(pids.shape[0], -1)
        for l, v in label_map.items():
            labels[pids==l] = v
        dset_labels[offset:offset_next] = labels

        polars = np.arccos(directions[:,1])
        azimuths = np.arctan2(directions[:,2], directions[:,0])
        dset_angles[offset:offset_next,:] = np.hstack((polars.reshape(-1,1),azimuths.reshape(-1,1)))

        for i, (pids, energies, starts, stops) in enumerate(zip(track_pid, track_energy,track_start_position, track_stop_position)):
            muons_above_threshold = (np.abs(pids) == 13) & (energies > 166)
            electrons_above_threshold = (np.abs(pids) == 11) & (energies > 2)
            gammas_above_threshold = (np.abs(pids) == 22) & (energies > 2)
            above_threshold = muons_above_threshold | electrons_above_threshold | gammas_above_threshold
            outside_tank = (np.linalg.norm(stops[:,(0,2)], axis=1) > config.radius) | (np.abs(stops[:, 1]) > config.half_height)
            dset_veto[offset+i] = np.any(above_threshold & outside_tank)
            end_energy_estimate = energies - np.linalg.norm(stops - starts, axis=1)*2
            muons_above_threshold = (np.abs(pids) == 13) & (end_energy_estimate > 166)
            electrons_above_threshold = (np.abs(pids) == 11) & (end_energy_estimate > 2)
            gammas_above_threshold = (np.abs(pids) == 22) & (end_energy_estimate > 2)
            above_threshold = muons_above_threshold | electrons_above_threshold | gammas_above_threshold
            dset_veto2[offset+i] = np.any(above_threshold & outside_tank)

        for i, (trigs, times, charges, pmts) in enumerate(zip(hit_triggers, hit_times, hit_charges, hit_pmts)):
            dset_event_hit_index[offset+i] = hit_offset
            hit_indices = np.where(trigs==event_triggers[i])[0]
            hit_offset_next += len(hit_indices)
            dset_hit_time[hit_offset:hit_offset_next] = times[hit_indices]
            dset_hit_charge[hit_offset:hit_offset_next] = charges[hit_indices]
            dset_hit_pmt[hit_offset:hit_offset_next] = pmts[hit_indices]
            hit_offset = hit_offset_next

        offset = offset_next
    f.close()
    print("saved", hit_offset, "hits in", offset, "good events (each with at least", min_hits, "hits)")
