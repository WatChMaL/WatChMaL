import matplotlib
from matplotlib import pyplot as plt


def combine_legends(ax):
    legends = [a.get_legend_handles_labels() for a in ax]
    return (l1 + l2 for l1, l2 in zip(*legends))


def plot_legend(ax):
    if isinstance(ax, matplotlib.axes.Axes):
        leg_params = ax.get_legend_handles_labels()
    else:
        leg_params = combine_legends(ax)
    leg_fig, leg_ax = plt.subplots(figsize=(1, 1))
    leg_ax.axis(False)
    leg_fig.set_tight_layout(False)
    leg_fig.legend(*leg_params, loc='center')
    return leg_fig, leg_ax


def plot_training_progression(train_epoch, train_loss, val_epoch, val_loss, val_best=None, y_lim=None,
                              fig_size=None, title=None, legend='center right'):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title(title)
    ax.plot(train_epoch, train_loss, lw=2, label='Train loss', color='b', alpha=0.3)
    ax.plot(val_epoch, val_loss, lw=2, label='Validation loss', color='b')
    if val_best is not None:
        ax.plot(val_epoch[val_best], val_loss[val_best], lw=0, marker='o', label='Best validation accuracy',
                 color='darkblue')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_ylabel("Loss", c='b')
    ax.set_xlabel("Epoch")
    if legend:
        ax.legend(loc=legend)
    return fig, ax
