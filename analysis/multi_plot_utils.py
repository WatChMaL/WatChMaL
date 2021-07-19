"""
Utils for plotting model performance for multiple runs at the same time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

from sklearn.metrics import auc

from WatChMaL.analysis.plot_utils import compute_roc, plot_roc, disp_learn_hist

def multi_disp_learn_hist(locations, losslim=None, show=True, titles=None, best_only=False, leg_font=10, title_font=10, xmax=None):
    '''
    Plot a grid of learning histories

    Args:
        locations   ... list of paths to directories of training dumps
        losslim     ... limit of loss axis
        show        ... bool, whether to show the plot
        titles      ... list of titles for each plot in the grid
        best_only   ... bool, whether to plot only the points where best model was saved
        leg_font    ... legend font size
        title_font  ... title font size
        xmax        ... maximum value on x-axis
    author: Calum Macdonald
    June 2020
    '''
    ncols = 1
    nrows = math.ceil(len(locations))

    if nrows==1 and ncols==1:
        fig = plt.figure(facecolor='w',figsize=(12,12))
    else:
        fig = plt.figure(facecolor='w',figsize=(12, nrows*4*3))
    
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    axes = []

    for i, location in enumerate(locations):
        print("i: ", i)
        if i == 0:
            ax1 = fig.add_subplot(gs[i], facecolor='w')
        else:
            ax1 = fig.add_subplot(gs[i], facecolor='w', sharey=axes[0])
        disp_learn_hist(location, axis=ax1, title=titles[i], losslim=losslim, show=False)
        axes.append(ax1)
    
    gs.tight_layout(fig)
    return fig

def multi_compute_roc(softmax_out_val_list, labels_val_list, true_label, false_label):
    """
    Call compute_roc on multiple sets of data

    Args:
        softmax_out_val_list    ... list of arrays of softmax outputs
        labels_val_list         ... list of 1D arrays of actual labels
        true_label              ... label of class to be used as true binary label
        false_label             ... label of class to be used as false binary label

    """
    fprs, tprs, thrs = [], [], []
    for softmax_out_val, labels_val in zip(softmax_out_val_list, labels_val_list):
        fpr, tpr, thr = compute_roc(softmax_out_val, labels_val, true_label, false_label)
        fprs.append(fpr)
        tprs.append(tpr)
        thrs.append(thr)

    return fprs, tprs, thrs

def multi_plot_roc(fprs, tprs, thrs, true_label_name, false_label_name, fig_list=None, xlims=None, ylims=None, axes=None, linestyles=None, linecolors=None, plot_labels=None, show=False):
    '''
    Plot multiple ROC curves of background rejection vs signal efficiency. Can plot 'rejection' (1/fpr) or 'fraction' (tpr).

    Args:
        fprs, tprs, thrs        ... list of false positive rate, list of true positive rate, list of thresholds used to compute scores
        true_label_name         ... name of class to be used as true binary label
        false_label_name        ... name of class to be used as false binary label
        fig_list                ... list of indexes of ROC curves to plot
        xlims                   ... xlims to apply to plots
        ylims                   ... ylims to apply to plots
        axes                    ... axes to plot on
        linestyle, linecolor    ... lists of line styles and colors
        plot_labels             ... list of strings to use in title of plots
        show                    ... if true then display figures, otherwise return figures
    '''
    rejections = [1.0/(fpr+1e-10) for fpr in fprs]
    AUCs = [auc(fpr,tpr) for fpr, tpr in zip(fprs, tprs)]

    num_panes = len(fig_list)
    fig, axes = plt.subplots(num_panes, 1, figsize=(12,8*num_panes))
    if num_panes > 1:
        fig.suptitle("ROC for {} vs {}".format(true_label_name, false_label_name), fontweight='bold',fontsize=32)

    # Needed for 1 plot case
    axes = np.array(axes).reshape(-1)

    for idx, fpr, tpr, thr in zip(range(len(fprs)), fprs, tprs, thrs):
        figs = plot_roc(fpr, tpr, thr, 
        true_label_name, false_label_name, 
        axes=axes, fig_list=fig_list, xlims=xlims, ylims=ylims,
        linestyle=linestyles[idx]  if linestyles is not None else None,
        linecolor=linecolors[idx] if linecolors is not None else None,
        plot_label=plot_labels[idx] if plot_labels is not None else None,
        show=False)

    return figs
