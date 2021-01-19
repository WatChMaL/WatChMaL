"""
Class for plotting hit data and generated GAN data for CNN mPMT datasets
"""

from analysis.event_analysis.EventPlotter import EventPlotter
from watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset import pmts_per_mpmt

class GANEventPlotter(EventPlotter):
    def __init__(self, h5_path, mpmt_positions_file, geo_path):
        super().__init__(h5_path, mpmt_positions_file, geo_path)
    
    def reconstruct_gan_data(self, image_data):
        """
        Retrieve PMT hit data from generated image data
        """
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
    
    def display_gan_image(self, image_data, ax=None):
        """
        Plot image produced by generator
        """
        hit_pmts, hit_data = self.reconstruct_gan_data(image_data)
        return self.plot_data(hit_pmts, hit_data, cutrange=[-1,1], ax=ax)
    
    def load_image_batches(self, path):
        """
        Load image batches produced by generator
        """
        return [np.load(fname, allow_pickle=True)['gen_imgs'] for fname in glob.glob(os.path.join(path,'imgs/*'))]
    
    def display_batch(self, image_batch, axes):
        """
        Display collapsed image batches in higher resolution plt format
        """
        ims = []
        for idx, ax in enumerate(axes.reshape(-1)):
            image_data = image_batch[idx]
            print(image_batch.shape)
            im = self.display_gan_image(image_data, ax=ax)
            ims.append(im)
            if idx > 8:
                break
        # return ims
