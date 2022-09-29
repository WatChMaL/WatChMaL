"""
Tools for event displays from CNN mPMT dataset
"""
import numpy as np
from analysis.event_display.event_display import plot_event_2d, plot_event_3d
from watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset import CNNmPMTDataset
from matplotlib.pyplot import cm
import torch


def channel_position_offset(channel):
    """
    Calculate offset in plotting coordinates for channel of PMT within mPMT.

    Parameters
    ----------
    channel: array_like of float
        array of channel IDs or PMT IDs

    Returns
    -------
    np.ndarray:
        Array of (y, x) coordinate offsets
    """
    channel = channel % 19
    theta = (channel < 12)*2*np.pi*channel/12 + ((channel >= 12) & (channel < 18))*2*np.pi*(channel-12)/6
    radius = 0.2*(channel < 18) + 0.2*(channel < 12)
    position = np.column_stack((radius*np.sin(theta), radius*np.cos(theta)))
    return position


def coordinates_from_data(data):
    """
    Calculate plotting coordinates for each element of CNN data, where each element of `data` contains a PMT's data.
    The actual values in `data` don't matter, it just takes the data tensor that has dimensions of
    (channel, image row, image column) and returns plotting coordinates for each element of the flattened data array.
    Plotting coordinates returned correspond to [x, y] coordinates for each PMT, including offsets for the PMT's
    position in the mPMT.

    Parameters
    ----------
    data: array_like
        Array of PMT data formatted for use in CNN, i.e. with dimensions of (channel, row, column)

    Returns
    -------
    coordinates: np.ndarray
        Coordinates for plotting the data
    """
    indices = np.indices(data.shape)
    channels = indices[0].flatten()
    coordinates = indices[[2, 1]].reshape(2, -1).astype(np.float64).T
    coordinates += channel_position_offset(channels)
    return coordinates


class CNNmPMTEventDisplay(CNNmPMTDataset):
    """
    This class extends the CNNmPMTDataset class to provide event display functionality.
    """
    def plot_data_2d(self, data, transformations=None, **kwargs):
        """
        Plots CNN mPMT data as a 2D event-display-like image.

        Parameters
        ----------
        data : array_like
            Array of PMT data formatted for use in CNN, i.e. with dimensions of (channels, x, y)
        transformations : function or str or sequence of function or str, optional
            Transformation function, or the name of a method of the dataset, or a sequence of functions or method names
            to apply to the data, such as those used for augmentation.
        kwargs : optional
            Additional arguments to pass to `analysis.event_display.plot_event_2d`.
            Valid arguments are:
            fig_width : scalar, optional
                Width of the figure
            title : str, default: None
                Title of the plot
            style : str, optional
                matplotlib style
            color_label: str, default: "Charge"
                Label to print next to the color scale
            color_map : str or Colormap, default: plt.cm.plasma
                Color map to use when plotting the data
            color_norm : matplotlib.colors.Normalize, optional
                Normalization to apply to color scale, by default uses log scaling
            show_zero : bool, default: false
                If false, zero data is drawn as the background color

        Returns
        -------
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
        """
        rows = self.mpmt_positions[:, 0]
        columns = self.mpmt_positions[:, 1]
        data = torch.Tensor(data)
        mpmt_locations = torch.zeros_like(data, dtype=bool)  # fill a data-like array with False
        mpmt_locations[18, rows, columns] = True  # replace channel 18 with True where there is an actual mPMT
        if transformations is not None:
            data = self.apply_transformation(transformations, data)
            mpmt_locations = self.apply_transformation(transformations, mpmt_locations)
        coordinates = coordinates_from_data(data)  # coordinates corresponding to each element of the data array
        data_nan = np.full_like(data, np.nan)  # fill an array with nan for positions where there's no actual PMTs
        data_nan[:, mpmt_locations[18]] = data[:, mpmt_locations[18]]  # replace the nans with the data where there is a PMT
        mpmt_coordinates = coordinates[mpmt_locations.flatten()]  # the coordinates of where the actual mPMTs are
        return plot_event_2d(data_nan.flatten(), coordinates, mpmt_coordinates, **kwargs)

    def plot_event_2d(self, event, transformations=None, **kwargs):
        """
        Plots an event as a 2D event-display-like image.

        Parameters
        ----------
        event : int
            index of the event to plot
        transformations : function or str or sequence of function or str, optional
            Transformation function, or the name of a method of this class, or a sequence of functions or method names
            to apply to the event data, such as those used for augmentation.
        kwargs : optional
            Additional arguments to pass to `analysis.event_display.plot_event_2d`
            Valid arguments are:
            fig_width : scalar, optional
                Width of the figure
            title : str, default: None
                Title of the plot
            style : str, optional
                matplotlib style
            color_label: str, default: "Charge"
                Label to print next to the color scale
            color_map : str or Colormap, default: plt.cm.plasma
                Color map to use when plotting the data
            color_norm : matplotlib.colors.Normalize, optional
                Normalization to apply to color scale, by default uses log scaling
            show_zero : bool, default: false
                If false, zero data is drawn as the background color

        Returns
        -------
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
        """
        data = self[event]['data']
        return self.plot_data_2d(data, transformations=transformations, **kwargs)

    def apply_transformation(self, transformation, data):
        """
        Apply a transformation or sequence of transformations to data.

        Parameters
        ----------
        transformation : function or str or sequence of function or str, optional
            Transformation function, or the name of a method of this class, or a sequence of functions or method names
            to apply to the event data, such as those used for augmentation.
        data : array_like
            Array of PMT data formatted for use in CNN, i.e. with dimensions of (channels, x, y)

        Returns
        -------
        np.ndarray
            transformed data
        """
        if isinstance(transformation, str):
            transformation = getattr(self, transformation)
        if callable(transformation):
            return transformation(data)
        else:
            for t in transformation:
                data = self.apply_transformation(t, data)
            return data

    def plot_event_3d(self, event, geometry_file_path, **kwargs):
        """
        Plots an event as a 3D event-display-like image.

        Parameters
        ----------
        event : int
            index of the event to plot
        geometry_file_path : str
            path to the file containing 3D geometry of detector
        kwargs : optional
            Additional arguments to pass to `analysis.event_display.plot_event_3d`
            Valid arguments are:
            fig_size : (float, float), optional
                Size of the figure
            zoom : float, default: 1.4
                Zoom factor to enlarge the 3D drawing
            title : str, default: None
                Title of the plot
            style : str, optional
                matplotlib style
            color_label: str, default: "Charge"
                Label to print next to the color scale
            color_map : str or Colormap, default: plt.cm.plasma
                Color map to use when plotting the data
            color_norm : matplotlib.colors.Normalize, optional
                Normalization to apply to color scale, by default uses log scaling
            vertical_axis : str, default: 'y'
                Vertical axis
            view_azimuth : float, default: -120
                Azimuthal angle of 3D camera view
            view_elevation : float, default: 30
                Elevation angle of 3D camera view
            show_zero : bool, default: false
                If false, zero data is drawn as the background color

        Returns
        -------
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
        """
        self.__getitem__(event)
        geo_file = np.load(geometry_file_path)
        pmt_coordinates = geo_file['position']
        data_coordinates = pmt_coordinates[self.event_hit_pmts, :]
        unhit_coordinates = np.delete(pmt_coordinates, self.event_hit_pmts, axis=0)
        data = np.array(self.event_hit_charges)
        return plot_event_3d(data, data_coordinates, unhit_coordinates, **kwargs)

    def plot_geometry(self, geometry_file_path, plot=('x', 'y', 'z'), view='2d', **kwargs):
        """
        Produces plots of detector geometry in 2D or 3D.

        Parameters
        ----------
        geometry_file_path : str
            path to the file containing 3D geometry of detector
        plot : str or sequence of str, optional
            The quantity to use to color the points corresponding to PMTs in the plot(s). This can be either as one of or a
            sequence of the following values. If a sequence is given, then a plot is produced for each element.
            ====== =======================================================================================
            `plot` color of each PMT marker in the plot
            ====== =======================================================================================
            '1'    The value 1 for all PMTs, to get the same colour at each PMT
            'i'    The ID of the PMT
            'x'    The x-coordinate of the PMT's 3D position
            'y'    The y-coordinate of the PMT's 3D position
            'z'    The z-coordinate of the PMT's 3D position
            'dx'   The x-coordinate of the unit vector of the PMT's normal direction
            'dy'   The y-coordinate of the unit vector of the PMT's normal direction
            'dz'   The z-coordinate of the unit vector of the PMT's normal direction
            'mx'   The x-coordinate of the PMT's 3D position relative to the centre PMT of its mPMT module
            'my'   The y-coordinate of the PMT's 3D position relative to the centre PMT of its mPMT module
            'mz'   The z-coordinate of the PMT's 3D position relative to the centre PMT of its mPMT module
            'ir'   The row of the PMT's 2D position in the CNN image
            'ic'   The column of the PMT's 2D position in the CNN image
            'ch'   The channel of the PMT within the mPMT
            ====== =======================================================================================
            By default, plots are produced for 'x', 'y' and 'z'.
        view : {'2d', '3d'}
            Whether to plot in 2D or 3D event display
        kwargs : optional
            Additional arguments to pass to the plotting function. See documentation for `plot_event_2d` and
            `plot_event_3d` for details.

        Returns
        -------
        figs: Tuple of matplotlib.figure.Figure
        axes: Tuple of matplotlib.axes.Axes
        """
        if isinstance(plot, str):
            plot = [plot]
        geo_file = np.load(geometry_file_path)
        pmt_coordinates = geo_file['position']
        pmt_directions = geo_file['orientation']
        pmt_ids = np.arange(pmt_coordinates.shape[0])
        mpmts = pmt_ids//19
        center_pmt_ids = mpmts*19+18  # the index of the centre PMT in the same module
        pmt_offset_positions = pmt_coordinates[pmt_ids]-pmt_coordinates[center_pmt_ids]
        pmt_mpmts = pmt_ids % 19
        data_map = {
            '1': np.ones(pmt_ids.shape),
            'i': pmt_ids,
            'x': pmt_coordinates[:, 0],
            'y': pmt_coordinates[:, 1],
            'z': pmt_coordinates[:, 2],
            'dx': pmt_directions[:, 0],
            'dy': pmt_directions[:, 1],
            'dz': pmt_directions[:, 2],
            'mx': pmt_offset_positions[:, 0],
            'my': pmt_offset_positions[:, 1],
            'mz': pmt_offset_positions[:, 2],
            'ir': self.mpmt_positions[mpmts, 0],
            'ic': self.mpmt_positions[mpmts, 1],
            'ch': pmt_mpmts,
        }
        title_map = {
            '1': None,
            'i': "PMT ID",
            'x': "PMT x-coordinate",
            'y': "PMT y-coordinate",
            'z': "PMT z-coordinate",
            'dx': "x-coordinate of PMT direction",
            'dy': "y-coordinate of PMT direction",
            'dz': "z-coordinate of PMT direction",
            'mx': "PMT x-coordinate relative to mPMT module",
            'my': "PMT y-coordinate relative to mPMT module",
            'mz': "PMT z-coordinate relative to mPMT module",
            'ir': "PMT row in CNN image",
            'ic': "PMT column in CNN image",
            'ch': "PMT channel in mPMT",
        }
        figs, axes = [], []
        for p in plot:
            if p in ['i', 'r', 'c', 'ch']:
                color_map = cm.turbo
            elif p == '1':
                color_map = cm.Greys
            else:
                color_map = cm.bwr
            args = {
                'color_map': color_map,
                'title': title_map[p],
                'color_norm': None,
                'show_zero': True,
                **kwargs
            }
            if view == '3d':
                data = data_map[p]
                fig, ax = plot_event_3d(data, pmt_coordinates, **args)
            else:
                data = self.process_data(pmt_ids, data_map[p])
                fig, ax = self.plot_data_2d(data, **args)
            figs.append(fig)
            axes.append(ax)
        return figs, axes

    def plot_geometry_2d(self, geometry_file_path, plot=('x', 'y', 'z'), **kwargs):
        """
        Calls `CNNmPMTEventDisplay.plot_geometry` with `view='2d'`.
        See documentation of `CNNmPMTEventDisplay.plot_geometry`.
        """
        return self.plot_geometry(geometry_file_path, plot, view='2d', **kwargs)

    def plot_geometry_3d(self, geometry_file_path, plot=('x', 'y', 'z'), **kwargs):
        """
        Calls `CNNmPMTEventDisplay.plot_geometry` with `view='3d'`.
        See documentation of `CNNmPMTEventDisplay.plot_geometry`.
        """
        return self.plot_geometry(geometry_file_path, plot, view='3d', **kwargs)
