import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

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

    ncols = 1#len(locations) if len(locations) < 3 else 3
    nrows = math.ceil(len(locations))#/3)
    if nrows==1 and ncols==1: fig = plt.figure(facecolor='w',figsize=(12,12))
    else: fig = plt.figure(facecolor='w',figsize=(12,nrows*4*3))
    gs = gridspec.GridSpec(nrows,ncols,figure=fig)
    axes = []
    for i,location in enumerate(locations):
        train_log=location+'/log_train.csv'
        val_log=location+'/log_val.csv'        
        train_log_csv = pd.read_csv(train_log)
        val_log_csv  = pd.read_csv(val_log)
        
        if best_only:
            best_idxs = [0]
            best_epoch=0
            best_loss = val_log_csv.loss[0]
            for idx,loss in enumerate(val_log_csv.loss):
                if loss < best_loss: 
                    best_loss=loss
                    best_idxs.append(idx)
                    best_epoch=val_log_csv.epoch[idx]
            val_log_csv = val_log_csv.loc[best_idxs]
            if titles is not None:
                titles[i] = titles[i] + ", Best Val Loss ={loss:.4f}@Ep.{epoch:.2f}".format(loss=best_loss,epoch=best_epoch)
                
        ax1=fig.add_subplot(gs[i],facecolor='w') if i ==0 else fig.add_subplot(gs[i],facecolor='w',sharey=axes[0])
        if xmax is None:
            xmax = train_log_csv.epoch.max()
        ax1.set_xlim(0,xmax)
        axes.append(ax1)
        line11 = ax1.plot(train_log_csv.epoch, train_log_csv.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
        line12 = ax1.plot(val_log_csv.epoch, val_log_csv.loss, marker='o', markersize=3, linestyle='', label='Validation loss', color='blue')
        if losslim is not None:
            ax1.set_ylim(None,losslim)
        if titles is not None:
            ax1.set_title(titles[i],size=title_font)
        ax2 = ax1.twinx()
        try:
            line21 = ax2.plot(train_log_csv.epoch, train_log_csv.accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
            line22 = ax2.plot(val_log_csv.epoch, val_log_csv.accuracy, marker='o', markersize=3, linestyle='', label='Validation accuracy', color='red')
        except:
            line21 = ax2.plot(train_log_csv.epoch, train_log_csv.acc, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
            line22 = ax2.plot(val_log_csv.epoch, val_log_csv.acc, marker='o', markersize=3, linestyle='', label='Validation accuracy', color='red')            

        ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
        ax1.tick_params('x',colors='black',labelsize=18)
        ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
        ax1.tick_params('y',colors='b',labelsize=18)

        ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
        ax2.tick_params('y',colors='r',labelsize=18)
        ax2.set_ylim(0.,1.05)

        lines  = line11 + line12 + line21 + line22
        labels = [l.get_label() for l in lines]
        leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1,prop={'size':leg_font})
        leg_frame = leg.get_frame()
        leg_frame.set_facecolor('white')
    gs.tight_layout(fig)
    return fig

def multi_plot_roc(data, metric, pos_neg_labels, plot_labels = None, png_name=None,title='ROC Curve', annotate=True,ax=None, linestyle=None, leg_loc=None, xlabel=None,ylabel=None,legend_label_dict=None):
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
    return