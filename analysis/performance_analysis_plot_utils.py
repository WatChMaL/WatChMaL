"""
Utils for analyzing performance depending on specific variables
"""

import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from textwrap import wrap

from WatChMaL.analysis.performance_analysis_utils import compute_fixed_operating_performance, plot_fixed_operating_performance, compute_multi_var_fixed_operating_performance, plot_multi_var_fixed_operating_performance
from WatChMaL.analysis.performance_analysis_utils import compute_pion_fixed_operating_performance, compute_pion_multi_var_fixed_operating_performance

# ========================================================================
# Constants

# TODO: fix cmap selection
cmaps = {'r':'Reds', 'g':'Greens', 'b':'Blues', 'purple':'Purples', 'orange':'Oranges'}
# TODO: move label from global to param
label_size = 14

# ========================================================================
# Single Variable Plotting Functions

def plot_fitqun_binned_performance(axes=None, eff_yrange=[0.5, 1.05], rej_yrange=[0, 0.006], **kwargs):
    '''
        Purpose: Re-create official FiTQun plots.
    '''
    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(18,9), facecolor='w')

    plot_momentum_binned_performance(metric='efficiency', yrange=eff_yrange, xrange=[0, 1000], plot_bins=None, ax=axes[0], **kwargs)

    plot_momentum_binned_performance(metric='fpr', yrange=rej_yrange, xrange=[0, 1000], plot_bins=None, ax=axes[1], **kwargs)


def plot_single_var_binned_performance(scores_list, labels_list, reconstructed_momentum, plot_binning_features, fpr_fixed_point, index_dict, recons_mom_bin_size=50, plot_bins=None, 
                            ax=None,marker='o--',colors=None,title_note='',metric='efficiency',yrange=None, xrange=None, names=None,
                            plot_bin_label=None, plot_bin_units=None, desired_labels=['$e$','$\mu$'], efficiency_correction_factor=1., rejection_correction_factor=1., show_x_err=False, show_legend=True):
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(9,5), facecolor='w')

    for idx, _ in enumerate(scores_list):
        bin_centers, bin_metrics, yerr = compute_fixed_operating_performance(scores_list[idx], labels_list[idx], reconstructed_momentum, plot_binning_features,
                                                                             fpr_fixed_point, index_dict, recons_mom_bin_size, plot_bins, metric, desired_labels, efficiency_correction_factor, rejection_correction_factor)
        plot_fixed_operating_performance(bin_centers, bin_metrics, yerr, 
                                    marker, colors[idx], names[idx], 'Reconstructed Momentum', plot_bin_label, plot_bin_units,
                                    fpr_fixed_point, title_note, metric, 
                                    yrange, xrange, ax, show_x_err=show_x_err, desired_labels=desired_labels, show_legend=show_legend)

    return ax


def plot_momentum_binned_performance(**kwargs):
    reconstructed_momentum = kwargs['reconstructed_momentum']
    recons_mom_bin_size    = kwargs['recons_mom_bin_size']
    plot_bins              = kwargs['plot_bins']
    
    if plot_bins is None:
        kwargs['plot_bins'] = np.array([0. + recons_mom_bin_size * i for i in range(math.ceil(np.max(reconstructed_momentum)/recons_mom_bin_size))])
    
    return plot_single_var_binned_performance(plot_bin_label='Reconstructed Momentum', plot_bin_units='MeV/c', plot_binning_features=reconstructed_momentum, **kwargs)


def plot_true_momentum_binned_performance(momentum_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='True Momentum',  plot_bin_units='MeV/c', plot_binning_features=momentum_features, **kwargs)


def plot_energy_binned_performance(energy_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='Energy', plot_bin_units='MeV/$c^2$', plot_binning_features=energy_features, **kwargs)


def plot_to_wall_binned_performance(to_wall_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='To Wall',  plot_bin_units='cm', plot_binning_features=to_wall_features, **kwargs)


def plot_zenith_binned_performance(zenith_features, **kwargs):
    ax = plot_single_var_binned_performance(plot_bin_label='Zenith',  plot_bin_units='Radians', plot_binning_features=zenith_features, **kwargs)
    plot_tank_corners(ax)
    ax.legend()
    return ax


def plot_azimuth_binned_performance(azimuth_features, **kwargs):
    ax = plot_single_var_binned_performance(plot_bin_label='Azimuth',  plot_bin_units='Radians', plot_binning_features=azimuth_features, **kwargs)
    plot_barrel_cut(ax)
    ax.legend()
    return ax


def plot_z_binned_performance(to_wall_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='z',  plot_bin_units='cm', plot_binning_features=to_wall_features, **kwargs)


def plot_radius_binned_performance(to_wall_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='Radius',  plot_bin_units='cm', plot_binning_features=to_wall_features, **kwargs)


# ========================================================================
# Multiple Variable Plotting Functions

def plot_multi_var_binned_performance(
                            scores_list, labels_list, 
                            reconstructed_momentum, 
                            binning_features, 
                            binning_bin_size,
                            plot_binning_features,
                            index_dict, 
                            fixed_bin_label        = 'Reconstructed Momentum',
                            binning_bin_label      = 'Azimuth',
                            plot_bin_label         = 'Zenith',
                            fixed_bin_units        = 'Mev/c',
                            binning_bin_units      = 'Radians',
                            plot_bin_units         = 'Radians',
                            fpr_fixed_point= 0.005, 
                            recons_mom_bin_size=50, plot_bins=None, 
                            axes=None,marker='o--',colors=None,
                            desired_labels=['$e$','$\mu$'],
                            title_note = '',
                            metric='efficiency',yrange=None, xrange=None, names=None,
                            efficiency_correction_factor=1., rejection_correction_factor=1.,):
    
    title_note = ' in Bins of {}'.format(binning_bin_label) + title_note

    if axes is None:
        fig, axes = plt.subplots(1,1,figsize=(9,5), facecolor='w')

    for idx, _ in enumerate(scores_list):
        cmap = plt.cm.get_cmap(cmaps[colors[idx]])
        all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins = compute_multi_var_fixed_operating_performance(
                                                                                        scores                 = scores_list[idx], 
                                                                                        labels                 = labels_list[idx],
                                                                                        fixed_binning_features = reconstructed_momentum, 
                                                                                        fixed_bin_size         = 50,
                                                                                        binning_features       = binning_features,
                                                                                        binning_bin_size       = binning_bin_size,
                                                                                        plot_binning_features  = plot_binning_features, 
                                                                                        plot_bins              = plot_bins,
                                                                                        index_dict             = index_dict, 
                                                                                        ignore_dict            = index_dict,
                                                                                        metric                 = 'efficiency',
                                                                                        fpr_fixed_point        = fpr_fixed_point,
                                                                                        desired_labels         = desired_labels,
                                                                                        efficiency_correction_factor=1., rejection_correction_factor=1.)
        
        plot_multi_var_fixed_operating_performance(all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins, 
                                                   marker, cmap, 
                                                   fixed_bin_label, binning_bin_label, plot_bin_label, 
                                                   fixed_bin_units, binning_bin_units, plot_bin_units,
                                                   fpr_fixed_point, title_note, metric, yrange, xrange, axes)

    legend_without_duplicate_labels(axes)
    return axes
    

def plot_to_wall_binned_in_energy(to_wall_features, **kwargs):
    return plot_multi_var_binned_performance(binning_bin_label='Energy', binning_bin_units='$MeV$', plot_bin_label='To Wall', plot_bin_units='cm', plot_binning_features=to_wall_features, **kwargs)


def plot_zenith_binned_in_azimuth(zenith_features, **kwargs):
    ax = plot_multi_var_binned_performance(binning_bin_label='Azimuth', binning_bin_units='Radians', plot_bin_label='Zenith',  plot_bin_units='Radians', plot_binning_features=zenith_features, **kwargs)
    plot_tank_corners(ax)
    legend_without_duplicate_labels(ax)
    return ax


def plot_azimuth_binned_in_zenith(azimuth_features, **kwargs):
    ax = plot_multi_var_binned_performance(binning_bin_label='Zenith',  binning_bin_units='Radians', plot_bin_label='Azimuth', plot_bin_units='Radians', plot_binning_features=azimuth_features, **kwargs)
    plot_barrel_cut(ax)
    legend_without_duplicate_labels(ax)
    return ax


def plot_binned_detector_angles(**kwargs):
    return plot_multi_var_binned_performance(binning_bin_label='Azimuth',  binning_bin_units='Radians', plot_bin_label='cos(Zenith)', plot_bin_units='', **kwargs)


def plot_binned_detector_volume(**kwargs):
    return plot_multi_var_binned_performance(binning_bin_label='z', binning_bin_units='cm', plot_bin_label='Radius', plot_bin_units='cm', **kwargs)


def plot_2D_hist(
                scores_list, labels_list, 
                reconstructed_momentum, 
                binning_features, 
                binning_bins,
                plot_binning_features,
                index_dict, 
                fixed_bin_label        = 'Reconstructed Momentum',
                binning_bin_label      = 'Azimuth',
                plot_bin_label         = 'Zenith',
                fixed_bin_units        = 'Mev/c',
                binning_bin_units      = 'Radians',
                plot_bin_units         = 'Radians',
                fpr_fixed_point= 0.005, 
                recons_mom_bin_size=50, plot_bins=None, 
                axes=None,marker='o--',colors=None,
                desired_labels=['$e$','$\mu$'],
                title_note = '',
                metric='efficiency',yrange=None, xrange=None, names=None):
    
    N = len(labels_list)
    fig, axes = plt.subplots(N,1,figsize=(9,10*N), facecolor='w')
    axes = axes.reshape(-1)

    _, binning_bins = np.histogram(binning_features, bins=binning_bins, range=(np.min(binning_features), np.max(binning_features)))
    binning_bin_size = binning_bins[1] - binning_bins[0]
    for idx, _ in enumerate(scores_list):
        ax = axes[idx]
        all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins = compute_multi_var_fixed_operating_performance(
                                                                                        scores                 = scores_list[idx], 
                                                                                        labels                 = labels_list[idx],
                                                                                        fixed_binning_features = reconstructed_momentum, 
                                                                                        fixed_bin_size         = 50,
                                                                                        binning_features       = binning_features,
                                                                                        binning_bin_size       = binning_bin_size,
                                                                                        plot_binning_features  = plot_binning_features, 
                                                                                        plot_bins              = plot_bins,
                                                                                        index_dict             = index_dict, 
                                                                                        ignore_dict            = index_dict,
                                                                                        #metric                 = 'efficiency',
                                                                                        fpr_fixed_point        = fpr_fixed_point,
                                                                                        desired_labels         = desired_labels,
                                                                                        efficiency_correction_factor = 1., 
                                                                                        rejection_correction_factor  = 1.)

        xedges, yedges = np.array(all_true_plotting_bins[0]), np.array(true_bins)
        mesh = np.array(all_bin_metrics)

        pc = ax.pcolormesh(xedges, yedges, mesh, cmap="viridis")
        fig.colorbar(pc, ax=ax)

        title_note = 'fpr_fixed_point'

        true_label  = desired_labels[0]
        false_label = desired_labels[1]

        if metric == 'efficiency':
            metric_name = '{} Signal Efficiency'.format(true_label) #'$e$- Signal Efficiency'
        else :
            metric_name = '{} Mis-ID Rate'.format(false_label) #'$\mu$- Mis-ID Rate'
        
        title = '{} vs {} and {} At {} Fixed Bin {} Mis-ID Rate'.format(metric_name, plot_bin_label, binning_bin_label, fixed_bin_label, false_label) 
        if title_note == 'fpr_fixed_point':
            title = title + ' {}%'.format(fpr_fixed_point*100) + ' for ' + names[idx]
        else:
            title = title + '{}'.format(title_note)
        title = "\n".join(wrap(title, 60))

        ax.set_xlabel("{} [{}]".format(binning_bin_label, binning_bin_units), fontsize=label_size)
        ax.set_ylabel("{} [{}]".format(plot_bin_label, plot_bin_units), fontsize=label_size)
        ax.set_title(title, fontsize=1.1*label_size)


def plot_2D_hist_to_wall_energy(**kwargs):
    plot_2D_hist(binning_bin_label='Energy', binning_bin_units='MeV/$c^2$', plot_bin_label='To Wall', plot_bin_units='cm', **kwargs)


def plot_2D_hist_zenith_azimuth(**kwargs):
    plot_2D_hist(binning_bin_label='Azimuth', binning_bin_units='Radians', plot_bin_label='Zenith', plot_bin_units='Radians', **kwargs)


def plot_varying_efficiency(fprs, plotting_function, **kwargs):
    '''
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(9,5), facecolor='w')
    '''
    colors = kwargs['colors']
    ax = kwargs['ax']

    model_cmaps = [plt.cm.get_cmap(cmaps[color]) for color in colors]
    range_val = np.linspace(0.4, 1, len(fprs))
    for idx, (fpr, range_val) in enumerate(zip(fprs, range_val)):
        kwargs['colors'] = [cmap(range_val) for cmap in model_cmaps]
        #kwargs['names']  = ['' for color in colors]
        plotting_function(fpr_fixed_point=fpr, **kwargs)
    
    legend_without_duplicate_labels(ax)

# ========================================================================
# Define equivalent plotting functions for pions

def plot_single_var_pion_binned_performance(
                                            scores, labels, 
                                            fixed_binning_features, fixed_bin_label, 
                                            plot_binning_features, plot_bin_label, plot_bin_units,
                                            p0, p1, pi0mass,
                                            fpr_fixed_point, index_dict, fixed_bin_size=50, plot_bins=20, 
                                            marker='o--',color='k', names=None, title_note='', metric='efficiency',yrange=None,xrange=None,
                                            ax = None, show_x_err=True, publication_style=False):

    bin_centers, bin_metrics, yerr = compute_pion_fixed_operating_performance(
                                     scores, labels, 
                                     fixed_binning_features,
                                     plot_binning_features,
                                     p0, p1, pi0mass,
                                     fpr_fixed_point, index_dict, fixed_bin_size, plot_bins, 
                                     metric)
    
    plot_fixed_operating_performance(bin_centers, bin_metrics, yerr, 
                                        marker, color, names[0],
                                        fixed_bin_label, plot_bin_label, 
                                        plot_bin_units=plot_bin_units,
                                        fpr_fixed_point=fpr_fixed_point, 
                                        title_note=title_note, 
                                        metric=metric, yrange=yrange, xrange=xrange, 
                                        ax=ax, publication_style=publication_style, show_x_err=show_x_err, desired_labels=['$e$','$\pi^0$'])
    
    #ax.legend(names)

def plot_pion_multi_var_binned_performance(
                            scores_list, labels_list, 
                            reconstructed_momentum, 
                            binning_features, 
                            binning_bin_size,
                            plot_binning_features,
                            p0, p1, pi0mass,
                            index_dict, 
                            fixed_bin_label        = 'Reconstructed Momentum',
                            binning_bin_label      = 'Azimuth',
                            plot_bin_label         = 'Zenith',
                            fpr_fixed_point= 0.005, 
                            recons_mom_bin_size=50, plot_bins=None, 
                            axes=None,marker='o--',colors=None,
                            desired_labels=['$e$','$\mu$'],
                            title_note = '',
                            metric='efficiency',yrange=None, xrange=None, names=None):
    
    title_note = ' in Bins of {}'.format(binning_bin_label) + title_note

    if axes is None:
        fig, axes = plt.subplots(1,1,figsize=(9,5), facecolor='w')

    for idx, _ in enumerate(scores_list):
        cmap = plt.cm.get_cmap(cmaps[colors[idx]])
        all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins = compute_pion_multi_var_fixed_operating_performance(
                                                                                        scores                 = scores_list[idx], 
                                                                                        labels                 = labels_list[idx],
                                                                                        fixed_binning_features = reconstructed_momentum, 
                                                                                        fixed_bin_size         = 50,
                                                                                        binning_features       = binning_features,
                                                                                        binning_bin_size       = binning_bin_size,
                                                                                        plot_binning_features  = plot_binning_features, 
                                                                                        plot_bins              = plot_bins,
                                                                                        p0                     = p0, 
                                                                                        p1                     = p1, 
                                                                                        pi0mass                = pi0mass,
                                                                                        index_dict             = index_dict, 
                                                                                        ignore_dict            = index_dict,
                                                                                        metric                 = 'efficiency',
                                                                                        fpr_fixed_point        = fpr_fixed_point,
                                                                                        desired_labels         = desired_labels)

        plot_multi_var_fixed_operating_performance(all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins, 
                                                   marker, cmap, 
                                                   fixed_bin_label, binning_bin_label, plot_bin_label, 
                                                   fpr_fixed_point, title_note, metric, yrange, xrange, axes, desired_labels)

    legend_without_duplicate_labels(axes)
    
# ========================================================================
# Define helper functions

def legend_without_duplicate_labels(ax):
    '''
    Merges legend elements with the same label (eliminates duplicate labels for models plotted with the same energy range)
    '''
    handles, labels = ax.get_legend_handles_labels()

    handle_dict = dict((k, []) for k in labels)
    for handle, label in zip(handles, labels):
        handle_dict[label].append(handle)

    new_handles = [tuple(h) for h in handle_dict.values()]
    new_labels  = handle_dict.keys()

    ax.legend(new_handles, new_labels, handlelength=5.0, handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)}, prop={'size': 16}, bbox_to_anchor=(1.05, 1), loc='upper left')


def plot_separate_legend(ax):
    legend_fig = plt.figure(figsize=(2, 1.25), facecolor='w')
    new_ax = legend_fig.add_subplot(111)
    new_ax.axis('off')

    handles, labels = ax.get_legend_handles_labels()

    new_ax.legend(handles, labels)
    

def plot_tank_corners(ax):
    handles, labels = ax.get_legend_handles_labels()

    if 'Tank Corner' not in labels:
        # arccos(R/sqrt(R^2 + half_height^2))
        tank_corner_angle = np.arccos(3/np.sqrt(3**2 + 4**2))
        ax.axvline(x=np.pi/2 - tank_corner_angle, c='k', linestyle='--', label='Tank Corner')
        ax.axvline(x=np.pi/2 + tank_corner_angle, c='k', linestyle='--')


def plot_barrel_cut(ax):
    handles, labels = ax.get_legend_handles_labels()

    if 'Barrel Cut' not in labels:
        ax.axvline(x=0, c='k', linestyle='--', label='Barrel Cut')