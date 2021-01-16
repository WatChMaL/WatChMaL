from analysis.event_analysis.EventPlotter import EventPlotter
from watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset import pmts_per_mpmt

class GANEventPlotter(EventPlotter):
    def __init__(self, h5_path, mpmt_positions_file, geo_path):
        super().__init__(h5_path, mpmt_positions_file, geo_path)
    
    def reconstruct_gan_data(self, image_data):
        # Silly off by one tube problem
        hit_pmts = self.tubes - 1

        # Get the mpmt that was hit
        hit_mpmts = hit_pmts // pmts_per_mpmt
        # Get the pmt within the module that was hit
        hit_pmt_in_modules = hit_pmts % pmts_per_mpmt

        hit_rows = self.dataset.mpmt_positions[hit_mpmts, 0]
        hit_cols = self.dataset.mpmt_positions[hit_mpmts, 1]

        hit_data = image_data[hit_pmt_in_modules, hit_rows, hit_cols]

        # TODO: decide what to do with negative values
        hit_data[hit_data < 0 ] = 0

        return hit_pmts, hit_data
