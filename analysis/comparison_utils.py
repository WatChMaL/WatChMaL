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

def multi_get_masked_data(data_list, cut_path, test_idxs, cut_list):
    cut_file = np.load(cut_path, allow_pickle=True) 

    cut_arrays = []
    for cut in cut_list:
        assert cut in cut_file.keys(), f"Error, {cut} has no associated cut file"
        cut_arrays.append(cut_file[cut][test_idxs])

    combined_cut_array = np.array(list(map(lambda x : 1 if 1 in x else 0,  list(zip(*cut_arrays)))))
    
    cut_idxs = np.where(combined_cut_array==1)[0]

    output_data_list = [np.delete(data, cut_idxs, 0) for data in data_list]
    return output_data_list

def collapse_test_output(softmaxes, labels, index_dict, predictions=None,ignore_type=None):
    '''
    Collapse gamma class into electron class to allow more equal comparison to FiTQun.
    Args:
        softmaxes                  ... 2d array of dimension (n,3) corresponding to softmax output
        labels                     ... 1d array of event labels, of length n, taking values in the set of values of 'index_dict'
        index_dict                 ... Dictionary with keys 'gamma','e','mu' pointing to the corresponding integer
                                       label taken by 'labels'
        predictions                ... 1d array of event type predictions, of length n, taking values in the 
                                       set of values of 'index_dict'   
        ignore_type                ... single string, name of event type to exclude                     
    '''
    if ignore_type is not None:
        keep_indices = np.where(labels!=index_dict[ignore_type])[0]
        softmaxes = softmaxes[keep_indices]
        labels = labels[keep_indices]
        if predictions is not None:
            predictions = predictions[keep_indices]

    new_labels = np.ones((softmaxes.shape[0]))*index_dict['$e$']
    new_softmaxes = np.zeros((labels.shape[0], 3))

    if predictions is not None:
        new_predictions = np.ones_like(predictions) * 1 #index_dict['$e$']
    
    for idx, label in enumerate(labels):
            if index_dict["$\mu$"] == label: new_labels[idx] = index_dict["$\mu$"]
            new_softmaxes[idx,:] = [0,softmaxes[idx][0] + softmaxes[idx][1], softmaxes[idx][2]]
            if predictions is not None:
                if predictions[idx]==index_dict['$\mu$']: new_predictions[idx] = index_dict['$\mu$']

    if predictions is not None: return new_softmaxes, new_labels, new_predictions
    
    return new_softmaxes, new_labels


def multi_collapse_test_output(output_softmax_list, actual_labels_list, label_dict, ignore_type=None):
    collapsed_class_scores_list, collapsed_class_labels_list = [],[]

    for softmaxes, labels in zip(output_softmax_list, actual_labels_list):
        collapsed_class_scores, collapsed_class_labels = collapse_test_output(softmaxes, labels, label_dict, ignore_type=ignore_type)
        collapsed_class_scores_list.append(collapsed_class_scores)
        collapsed_class_labels_list.append(collapsed_class_labels)

    return collapsed_class_scores_list, collapsed_class_labels_list