import matplotlib
from matplotlib import pyplot as plt
import analysis.utils.binning as bins


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


def plot_binned_values(ax, func, values, binning, selection=None, errors=False, x_errors=True, **plot_args):
    plot_args.setdefault('lw', 2)
    binned_values = bins.apply_binning(values, binning, selection)
    x = bins.bin_centres(binning[0])
    if errors:
        y_values, y_errors = func(binned_values, errors)
        x_errors = bins.bin_halfwidths(binning[0]) if x_errors else None
        plot_args.setdefault('marker', '')
        plot_args.setdefault('capsize', 4)
        plot_args.setdefault('capthick', 2)
        ax.errorbar(x, y_values, yerr=y_errors, xerr=x_errors, **plot_args)
    else:
        y = func(binned_values, errors)
        plot_args.setdefault('marker', 'o')
        ax.plot(x, y, **plot_args)