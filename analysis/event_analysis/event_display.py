"""
Tools for making event display style plots
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import copy


def plot_2d_event(data, positions, pmt_pos, dpi=300, title=None, style="default", color_map=plt.cm.jet, show_zero=False):
    """
    Plots 2D event display from PMT data (time or charge)

    Parameters
    ----------
    data : array_like
        Data to be plotted (e.g. PMT charges)
    positions : array_like
        2D (row, column) locations of where to plot the data
    pmt_pos : array_like
        2D (row, column) locations for circles to draw where PMTs or mPMTs are present
    dpi : int, default: 300
        Resolution of the figure
    title : str, default: None
        Title of the plot
    style : str, default: "default"
        matplotlib style
    color_map : str or Colormap, default: plt.cm.jet
        Color map to use when plotting the data
    show_zero : bool, default: false
        If false, zero data is drawn as the background color
    """

    if not show_zero:
        data[data == 0] = np.nan
    color_map = copy.copy(color_map)
    if style == "dark_background":
        edge_color = '0.15'
        color_map.set_bad(color='black')
    else:
        edge_color = '0.85'
        color_map.set_bad(color='white')
    axis_ranges = np.ptp(pmt_pos, axis=0)
    fig_size = (20, 16 * axis_ranges[0] / axis_ranges[1])
    pmt_circles = [Circle((pos[1], pos[0]), radius=0.47) for pos in pmt_pos]
    with plt.style.context(style):
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        ax.add_collection(PatchCollection(pmt_circles, facecolor='none', linewidths=1, edgecolors=edge_color))
        pmts = ax.scatter(positions[1], positions[0], c=data.flatten(), s=3, cmap=color_map)
        plt.colorbar(pmts)
    if title is not None:
        ax.set_title(title)


