import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

from sklearn.metrics import auc

from analysis.plot_utils import compute_roc, plot_roc, disp_learn_hist

def multi_disp_learn_hist(locations,losslim=None,show=True,titles=None,best_only=False,leg_font=10,title_font=10,xmax=None):
    '''
    Plots a grid of learning histories.
    Args:
        locations               ... list of paths to directories of training dumps
        losslim                 ... limit of loss axis
        show                    ... bool, whether to show the plot
        titles                  ... list of titles for each plot in the grid
        best_only               ... bool, whether to plot only the points where best model was saved
        leg_font                ... legend font size
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
            ax1 = fig.add_subplot(gs[i],facecolor='w')
        else:
            ax1 = fig.add_subplot(gs[i],facecolor='w',sharey=axes[0])
        disp_learn_hist(location, axis=ax1, show=False)
        axes.append(ax1)
    
    gs.tight_layout(fig)
    return fig

def multi_compute_roc(softmax_out_val_list, labels_val_list, true_label, false_label):
    # Compute ROC metrics
    fprs, tprs, thrs = [], [], []
    for softmax_out_val, labels_val in zip(softmax_out_val_list, labels_val_list):
        fpr, tpr, thr = compute_roc(softmax_out_val, labels_val, true_label, false_label)
        fprs.append(fpr)
        tprs.append(tpr)
        thrs.append(thr)

    return fprs, tprs, thrs

def multi_plot_roc(fprs, tprs, thrs, true_label_name, false_label_name, fig_list=None, xlims=None, ylims=None, axes=None, linestyles=None, linecolors=None, plot_labels=None, show=False):
    '''
    plot_multiple_ROC(data, metric, pos_neg_labels, plot_labels = None, png_name=None,title='ROC Curve', annotate=True,ax=None, linestyle=None, leg_loc=None, xlabel=None,ylabel=None,legend_label_dict=None)
    Plot multiple ROC curves of background rejection vs signal efficiency. Can plot 'rejection' (1/fpr) or 'fraction' (tpr).
    Args:
        data                ... tuple of (n false positive rates, n true positive rate, n thresholds) to plot rejection or 
                                (rejection fractions, true positive rates, false positive rates, thresholds) to plot rejection fraction.
        metric              ... string, name of metric to plot: ('rejection' or 'fraction')
        pos_neg_labels      ... array of one positive and one negative string label, or list of lists, with each list giving positive and negative label for
                                one dataset
        plot_labels         ... label for each run to display in legend
        png_name            ... name of image to save
        title               ... title of plot
        annotate            ... whether or not to include annotations of critical points for each curve, default True
        ax                  ... matplotlib.pyplot.axes on which to place plot
        linestyle           ... list of linestyles to use for each curve, can be '-', ':', '-.'
        leg_loc             ... location for legend, eg 'upper right' - vertical upper, center, lower, horizontal right left
        legend_label_dict   ... dictionary of display symbols for each string label, to use for displaying pretty characters in labels
    author: Calum Macdonald
    June 2020
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