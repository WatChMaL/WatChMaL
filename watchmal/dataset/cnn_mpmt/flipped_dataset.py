from watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset import CNNmPMTDataset

class flippedDataset(CNNmPMTDataset):
    def __init__(self, h5file, mpmt_positions_file, is_distributed, padding_type=None, transforms=None, collapse_arrays=False, pad=False):
        super().__init__(h5file, mpmt_positions_file, is_distributed, padding_type, transforms, collapse_arrays)

    
    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        data_dict['data'] = self.vertical_flip(data_dict['data'])
        return data_dict