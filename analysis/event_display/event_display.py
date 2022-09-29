"""
Tools for making event display style plots
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
import copy
from contextlib import nullcontext


def plot_event_2d(pmt_data, data_coordinates, pmt_coordinates, fig_width=None, title=None, style=None,
                  color_label=None, color_map=plt.cm.plasma, color_norm=colors.LogNorm(), show_zero=False):
    """
    Plots 2D event display from PMT data

    Parameters
    ----------
    pmt_data : array_like
        Data to be plotted (e.g. PMT charges)
    data_coordinates : array_like
        2D (x, y) locations of where to plot the data
    pmt_coordinates : array_like
        2D (x, y) locations for circles to draw where PMTs or mPMTs are present
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
    if not show_zero:
        pmt_data[pmt_data == 0] = np.nan
    color_map = copy.copy(color_map)
    if style == "dark_background":
        edge_color = '0.35'
        color_map.set_bad(color='black')
    else:
        edge_color = '0.85'
        color_map.set_bad(color='white')
    axis_ranges = np.ptp(pmt_coordinates, axis=0)
    if fig_width is None:
        fig_width = matplotlib.rcParams['figure.figsize'][0]
    scale = fig_width/20
    fig_size = (20*scale, 16*scale*axis_ranges[1]/axis_ranges[0])
    pmt_circles = [Circle((pos[0], pos[1]), radius=0.48) for pos in pmt_coordinates]
    with plt.style.context(style) if style else nullcontext():
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect(1)
        ax.add_collection(PatchCollection(pmt_circles, facecolor='none', linewidths=1*scale, edgecolors=edge_color))
        pmts = ax.scatter(data_coordinates[:, 0], data_coordinates[:, 1], c=pmt_data.flatten(), s=7*scale*scale, cmap=color_map, norm=color_norm)
        ax_min = np.min(pmt_coordinates, axis=0) - 1
        ax_max = np.max(pmt_coordinates, axis=0) + 1
        ax.set_xlim([ax_min[0], ax_max[0]])
        ax.set_ylim([ax_min[1], ax_max[1]])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        fig.colorbar(pmts, ax=ax, pad=0, label=color_label)
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_event_3d(pmt_data, data_coordinates, unhit_pmt_coordinates=None, fig_size=None, zoom=1.4, title=None,
                  style=None, color_label=None, color_map=plt.cm.plasma, color_norm=colors.LogNorm(),
                  show_zero=False, vertical_axis='y', view_azimuth=-90, view_elevation=30):
    """
    Plots 3D event display of colored data at each PMT in 3D space

    Parameters
    ----------
    pmt_data : array_like
        Data to be plotted (e.g. PMT charges)
    data_coordinates : array_like
        3D (x, y, z) locations of where to plot the data
    unhit_pmt_coordinates : array_like, optional
        3D (x, y, z) locations of all unhit PMTs
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
    if not show_zero:
        pmt_data[pmt_data == 0] = np.nan
    color_map = copy.copy(color_map)
    if style == "dark_background":
        unhit_color = '0.7'
        color_map.set_bad(color=unhit_color)
    else:
        unhit_color = '0.6'
        color_map.set_bad(color=unhit_color)

    with plt.style.context(style) if style else nullcontext():
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': '3d'})
        ax.view_init(elev=view_elevation, azim=view_azimuth, vertical_axis=vertical_axis)
        scale = np.min(fig.get_size_inches()/[15, 12])
        # It would be nice to plot circles oriented by their actual orientation, but this is difficult to do with 3D patches
        marker_size = 7*zoom*zoom*scale*scale
        ax_min = np.min(data_coordinates, axis=0)
        ax_max = np.max(data_coordinates, axis=0)
        if unhit_pmt_coordinates is not None:
            ax.scatter3D(*unhit_pmt_coordinates.T, c=unhit_color, marker='o', s=marker_size, alpha=0.1)
            ax_min = np.minimum(ax_min, np.min(unhit_pmt_coordinates, axis=0))
            ax_max = np.maximum(ax_max, np.max(unhit_pmt_coordinates, axis=0))
        hits = ax.scatter(*data_coordinates.T, c=pmt_data, marker='o', s=marker_size, alpha=0.8, cmap=color_map, norm=color_norm)
        ax_min /= zoom
        ax_max /= zoom
        ax.axes.set_xlim3d((ax_min[0], ax_max[0]))
        ax.axes.set_ylim3d((ax_min[1], ax_max[1]))
        ax.axes.set_zlim3d((ax_min[2], ax_max[2]))
        fig.colorbar(hits, ax=ax, pad=0, label=color_label)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if title is not None:
        ax.set_title(title)
    return fig, ax
