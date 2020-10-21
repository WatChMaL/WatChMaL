import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_multiple_ROC(data, metric, pos_neg_labels, plot_labels = None, png_name=None,title='ROC Curve', annotate=True,ax=None, linestyle=None, leg_loc=None, xlabel=None,ylabel=None,legend_label_dict=None):
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
    