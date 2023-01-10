"""
Tools for event displays from PointNet dataset
"""
import numpy as np
from analysis.event_display.event_display import plot_event_3d
from watchmal.dataset.pointnet.pointnet_dataset import PointNetDataset
from matplotlib.pyplot import cm


class PointNetEventDisplay(PointNetDataset):
    """
    This class extends the PointNetDataset class to provide event display functionality.
    """

    def plot_event_3d(self, event, data_channel=-1, **kwargs):
        """
        Plots an event as a 3D event-display-like image.

        Parameters
        ----------
        event : int
            index of the event to plot
        data_channel : int, default: -1
            The channel of the data used as the colours of the points. By default, channel -1 is used (i.e. the last
            channel of the data, usually charge or time). If set to None, each point is plotted with the same colour
            (gray by default unless color_map is set in kwargs).
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
        data = self.__getitem__(event)["data"]
        data_coordinates = data[:3, :].T
        if data_channel is None:
            data_values = np.ones_like(data[0, :])
            kwargs.setdefault("color_map", cm.bwr)
        else:
            data_values = data[data_channel, :]
        unhit_coordinates = np.delete(self.geo_positions, self.event_hit_pmts, axis=0)
        return plot_event_3d(data_values, data_coordinates, unhit_coordinates, **kwargs)
