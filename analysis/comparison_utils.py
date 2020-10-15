import numpy as np

def get_masked_data(data, cut_path, test_idxs, cut_list):
    cut_file = np.load(cut_path, allow_pickle=True) 

    cut_arrays = []
    for cut in cut_list:
        assert cut in cut_file.keys(), f"Error, {cut} has no associated cut file"
        cut_arrays.append(cut_file[cut][test_idxs])

    combined_cut_array=np.array(list(map(lambda x : 1 if 1 in x else 0,  list(zip(*cut_arrays)))))
    
    cut_idxs = np.where(combined_cut_array==1)[0]

    output_data = np.delete(data, cut_idxs, 0)

    return output_data