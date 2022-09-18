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


