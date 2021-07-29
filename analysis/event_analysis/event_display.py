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


def plot_3d_event(data, positions, pmt_positions, pmt_orientations, dpi=300, title=None, color_map=plt.cm.jet_r, vertical_axis='y'):
    """
    Plots 2D event display from PMT data (time or charge)

    Parameters
    ----------
    data : array_like
        Data to be plotted (e.g. PMT charges)
    positions : array_like
        3D (x, y, z) locations of where to plot the data
    pmt_positions : array_like
        3D (x, y, z) locations of all PMTs or mPMTs
    pmt_orientations : array_like
        2D (x, y, z) orientations of all PMTs or mPMTs
    dpi : int, default: 300
        Resolution of the figure
    title : str, default: None
        Title of the plot
    color_map : str or Colormap, default: plt.cm.jet
        Color map to use when plotting the data
    vertical_axis : str, default='y'
        Vertical axis
    """

    # for upcoming/new versions of matplotlib, this can be achieved more cleanly just with:
    # ax.view_init(vertical_axis=vertical_axis)
    # so we should change this once that matplotlib version is widely used
    if vertical_axis == 'y':
        positions = positions[:, (0, 2, 1)]
        pmt_positions = pmt_positions[:, (0, 2, 1)]
        pmt_orientations = pmt_orientations[:, (0, 2, 1)]
    elif vertical_axis == 'x':
        positions = positions[:, (2, 1, 0)]
        pmt_positions = pmt_positions[:, (2, 1, 0)]
        pmt_orientations = pmt_orientations[:, (2, 1, 0)]

    color_map = copy.copy(color_map)
    color_map.set_bad(color='white')

    fig = plt.figure(figsize=(20, 12), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    # It would be nice to plot circles oriented by their actual orientation, but this is difficult to do with 3D patches
    ax.scatter3D(*pmt_positions.T, c='gray', marker='o', s=2, alpha=0.1)
    hits = ax.scatter(*positions.T, c=data, s=2, cmap=color_map)
    plt.colorbar(hits)

    if title is not None:
        ax.set_title(title)