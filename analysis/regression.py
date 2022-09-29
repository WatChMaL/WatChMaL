import numpy as np
import tabulate
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from analysis.read import WatChMaLOutput
import analysis.utils.math as math
import analysis.utils.binning as bins
from analysis.utils.plotting import plot_binned_values


def plot_histograms(runs, quantity, selection=None, ax=None, fig_size=None, x_label="", y_label="", legend='best', **hist_args):
    """
    Plot overlaid histograms of results from a number of regression runs.

    Parameters
    ----------
    runs: sequence of RegressionRun
        Sequence of run results.
    quantity: str or callable
        Name of the attribute in each run that contains the quantities, or function that takes the run as its only
        argument and returns the quantities, to be histogrammed.
    selection: indexing expression, optional
        Selection of the values to be histogrammed (by default use all values).
    ax: matplotlib.axes.Axes
        Axes to draw the plot. If not provided, a new figure and axes is created.
    fig_size: (float, float), optional
        Figure size. Ignored if `ax` is provided.
    x_label: str, optional
        Label of the x-axis.
    y_label: str, optional
        Label of the y-axis.
    legend: str or None, optional
        Position of the legend, or None to have no legend. Attempts to find the best position by default.
    hist_args: optional
        Additional arguments to pass to the `hist` plotting function. Note that these may be overridden by arguments
        provided in `runs`.

    Returns
    -------
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    """
    hist_args.setdefault('bins', 200)
    hist_args.setdefault('density', True)
    hist_args.setdefault('histtype', 'step')
    hist_args.setdefault('lw', 2)
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.get_figure()
    for r in runs:
        if selection is None:
            run_selection = r.selection if selection is None else selection
        data = r.get_quantity(quantity)[run_selection].flatten()
        args = {**hist_args, **r.plot_args}
        ax.hist(data, **args)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.legend(loc=legend)
    return fig, ax


def plot_resolution_profile(runs, quantity, binning, selection=None, ax=None, fig_size=None, x_label="", y_label="",
                            legend='best', y_lim=None, **plot_args):
    """
    Plot binned resolutions for results from a set of regression runs. The quantity should be the name of an attribute
    that contains residuals (or similar quantity representing reconstruction errors), and the set of residuals are
    divided up into bins according to `binning`, before calculating the resolution (68th percentile of their absolute
    values) in each bin. A selection can be provided to use only a subset of all the values. The same binning and
    selection is applied to each run.

    Parameters
    ----------
    runs: sequence of RegressionRun
        Sequence of run results.
    quantity: str or callable
        Name of the attribute containing the reconstruction errors, or function that takes the run as its only argument
        and returns the reconstruction errors, whose average resolution will be plotted.
    binning: (np.ndarray, np.ndarray)
        Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
    selection: indexing expression, optional
        Selection of the values to use in calculating the resolutions (by default use all values).
    ax: matplotlib.axes.Axes
        Axes to draw the plot. If not provided, a new figure and axes is created.
    fig_size: (float, float), optional
        Figure size. Ignored if `ax` is provided.
    x_label: str, optional
        Label of the x-axis.
    y_label: str, optional
        Label of the y-axis.
    legend: str or None, optional
        Position of the legend, or None to have no legend. Attempts to find the best position by default.
    y_lim: (float, float), optional
        Limits of the y-axis.
    plot_args: optional
        Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
        provided in `runs`.

    Returns
    -------
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.get_figure()
    for r in runs:
        args = {**plot_args, **r.plot_args}
        r.plot_binned_resolution(quantity, ax, binning, selection, **args)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.legend(loc=legend)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig, ax


def plot_bias_profile(runs, quantity, binning, selection=None, ax=None, fig_size=None, x_label="", y_label="",
                      legend='best', y_lim=None, **plot_args):
    """
    Plot binned bias for results from a set of regression runs. The quantity should be the name of an attribute that
    contains residuals (or similar quantity representing reconstruction errors), or a function returning the residuals,
    and the set of residuals are divided up into bins according to `binning`, before calculating the resolution (68th
    percentile of their absolute values) in each bin. A selection can be provided to use only a subset of all the
    values. The same binning and selection is applied to each run.

    Parameters
    ----------
    runs: sequence of RegressionRun
        Sequence of run results.
    quantity: str or callable
        Name of the attribute containing the reconstruction errors, or function that takes the run as its only argument
        and returns the reconstruction errors, whose average resolution will be plotted.
    binning: (np.ndarray, np.ndarray)
        Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
    selection: indexing expression, optional
        Selection of the values to use in calculating the resolutions (by default use all values).
    ax: matplotlib.axes.Axes
        Axes to draw the plot. If not provided, a new figure and axes is created.
    fig_size: (float, float), optional
        Figure size. Ignored if `ax` is provided.
    x_label: str, optional
        Label of the x-axis.
    y_label: str, optional
        Label of the y-axis.
    legend: str or None, optional
        Position of the legend, or None to have no legend. Attempts to find the best position by default.
    y_lim: (float, float), optional
        Limits of the y-axis.
    plot_args: optional
        Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
        provided in `runs`.

    Returns
    -------
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.get_figure()
    for r in runs:
        args = {**plot_args, **r.plot_args}
        r.plot_binned_bias(quantity, ax, binning, selection, **args)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.legend(loc=legend)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig, ax


def tabulate_statistics(runs, quantities, stat_labels, statistic="resolution", selection=None, transpose=False,
                        **tabulate_args):
    """
    Return a table of summary statistics of quantities of runs.

    Parameters
    ----------
    runs: sequence of RegressionRun
        Sequence of run results.
    quantities: str or callable or sequence of str or sequence of callable
        Name(s) of the attribute(s) that contain the quantities, or function(s) that takes the run as its only argument
        and returns the quantities, whose statistics would be calculated.
    stat_labels: str or sequence of str
        Label(s) for the quantities / statistics being calculated, the same length as `quantities`.
    selection:
        Selection of the values to use in calculating the summary statistics (by default use all values).
    statistic: {'resolution', 'mean', callable, list of callable, list of str}
        The summary statistic to apply to the quantity. If callable, should be a function that takes the array_like of
        values and returns the summary statistic. If `resolution` (default) use the 68th percentile. If `mean` use the
        mean. If a list, should be the same length as `quantities` to specify the summary statistic of each quantity,
        with each element of the list one of the alternatives described above.
    transpose: bool
        If True, table rows correspond to each run and columns correspond to each quantity summary statistic. Otherwise,
        (default) rows correspond to summary statistics and columns correspond to runs.
    tabulate_args: optional
        Additional named arguments to pass to `tabulate.tabulate`. By default, set table format to `html` and float
        format to `.2f`.
    Returns
    -------
    str
        String representing the tabulated data
    """
    tabulate_args.setdefault('tablefmt', 'html')
    tabulate_args.setdefault('floatfmt', '.2f')
    if isinstance(quantities, str):
        quantities = [quantities]
    if isinstance(stat_labels, str):
        stat_labels = [stat_labels]
    run_labels = [r.run_label for r in runs]
    statistic_map = {
        "resolution": lambda x:  np.quantile(np.abs(x), 0.68),
        "mean": lambda x: np.mean(x),
    }
    if isinstance(statistic, str):
        functions = [statistic_map[statistic]]*len(quantities)
    elif callable(statistic):
        functions = [lambda x: [statistic(x)]]*len(quantities)
    else:
        functions = [statistic_map[stat] if isinstance(stat, str)
                     else (lambda x: stat(x))
                     for stat in statistic]
    data = []
    for f, q in zip(functions, quantities):
        data.append([f(r.get_quantity(q)[r.selection if selection is None else selection]) for r in runs])
    if transpose:
        data = list(zip(*data))
        return tabulate.tabulate(data, headers=stat_labels, showindex=run_labels, **tabulate_args)
    else:
        return tabulate.tabulate(data, headers=run_labels, showindex=stat_labels, **tabulate_args)


class RegressionRun(ABC):
    """
    Base class for regression results
    """
    def __init__(self, run_label, selection=None, **plot_args):
        """
        Create object to hold regression results

        Parameters
        ----------
        run_label: str
            Label to describe this set of results to use in plot legends, etc.
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            By default, use all results.
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results.
        """
        if selection is None:
            selection = ...
        self.selection = selection
        self.run_label = run_label
        plot_args['label'] = run_label
        self.plot_args = plot_args

    def get_quantity(self, quantity):
        """
        Return either an attribute of the run results given the name of the attribute, or a function of the run results.
        The attribute or function should return an array of values corresponding to a quantity whose resolution or mean
        is to be plotted, tabulated, etc.

        Parameters
        ----------
        quantity: str or callable
            Either a string representing the name of an attribute containing the array of values  to return, or a
            function that takes a RegressionRun as its only argument and returns an array of vales.

        Returns
        -------
        np.ndarray:
            One-dimensional array of values corresponding to the desired quantity
        """
        if isinstance(quantity, str):
            return getattr(self, quantity)
        elif callable(quantity):
            return quantity(self)
        else:
            raise TypeError("The quantity should be a string or function")

    def plot_binned_resolution(self, quantity, ax, binning, selection=None, errors=False, x_errors=True, **plot_args):
        """
        Plot binned resolutions of the results of a regression run on an existing set of axes. The quantity should be
        the name of an attribute that contains residuals (or similar quantity representing reconstruction errors), and
        the set of residuals are divided up into bins according to `binning`, before calculating the resolution (68th
        percentile of their absolute values) in each bin. A selection can be provided to use only a subset of all the
        values.

        Parameters
        ----------
        quantity: str or callable
            Name of the attribute containing the array of values, or function that takes the run as its only argument
            and returns the values, whose average resolution will be plotted.
        ax: matplotlib.axes.Axes
            Axes to draw the plot.
        binning: (np.ndarray, np.ndarray)
            Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
        selection: indexing expression, optional
            Selection of the values to use in calculating the resolutions (by default use all values).
        errors: bool, optional
            If True, plot error bars calculated as the standard deviation divided by sqrt(N) of the N values in the bin.
        x_errors: bool, optional
            If True, plot horizontal error bars corresponding to the width of the bin, only if `errors` is also True.
        plot_args: optional
            Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
            provided in `runs`.
        """
        if selection is None:
            selection = self.selection
        values = self.get_quantity(quantity)
        return plot_binned_values(ax, bins.binned_resolutions, values, binning, selection, errors, x_errors, **plot_args)

    def plot_binned_bias(self, quantity, ax, binning, selection=None, errors=False, x_errors=True, **plot_args):
        """
        Plot binned bias of the results of a regression run on an existing set of axes. The quantity should be the name
        of an attribute that contains residuals (or similar quantity representing reconstruction errors), or a function
        returning these residuals, and the set of residuals are divided up into bins according to `binning`, before
        calculating the bias (mean) in each bin. A selection can be provided to use only a subset of all the values.

        Parameters
        ----------
        quantity: str or callable
            Name of the attribute containing the array of values, or function that takes the run as its only argument
            and returns the values, whose average bias will be plotted.
        ax: matplotlib.axes.Axes
            Axes to draw the plot.
        binning: (np.ndarray, np.ndarray)
            Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
        selection: indexing expression, optional
            Selection of the values to use in calculating the resolutions (by default use all values).
        errors: bool, optional
            If True, plot error bars calculated as the standard deviation divided by sqrt(N) of the N values in the bin.
        x_errors: bool, optional
            If True, plot horizontal error bars corresponding to the width of the bin, only if `errors` is also True.
        plot_args: optional
            Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
            provided in `runs`.
        """
        if selection is None:
            selection = self.selection
        values = self.get_quantity(quantity)
        return plot_binned_values(ax, bins.binned_mean, values, binning, selection, errors, x_errors, **plot_args)


class MomentumPrediction(ABC):
    """
    Base class for predictions of the magnitude of a particle's momentum.

    The following attributes are calculated and provided if the required truth information is given:

    ===================================================================================================================
    `momentum_residuals`            true minus reconstructed momentum magnitude
    `energy_residuals`              true minus reconstructed energy, calculated from the momentum using the mass of the
                                    true particle type
    `momentum_fractional_errors`    `momentum_residuals` divided by the true momentum
    `energy_fractional_errors`      `energy_residuals` divided by the true energy
    ===================================================================================================================
    """
    def __init__(self, true_momenta=None, true_labels=None):
        """
        Create object to hold momentum predictions.

        Parameters
        ----------
        true_momenta: array_like of float, optional
            Array of true momenta, used in calculating residuals
        true_labels: int or array_like of int, optional
            PID label or array of PID labels, used to know the particle type to convert between momentum and energy
        """
        self.true_labels = true_labels
        self.true_momenta = true_momenta
        self.true_energies = None
        self.momentum_residuals = None
        self.energy_residuals = None
        if true_momenta is not None:
            self.momentum_residuals = self.momentum_prediction - self.true_momenta
            self.momentum_fractional_errors = self.momentum_residuals/self.true_momenta
            if true_labels is not None:
                self.true_energies = math.energy_from_momentum(true_momenta, true_labels)
                self.energy_residuals = self.energy_prediction - self.true_energies
                self.energy_fractional_errors = self.energy_residuals/self.true_energies

    @property
    @abstractmethod
    def momentum_prediction(self):
        """This attribute gives the predicted momenta"""

    @property
    def energy_prediction(self):
        """Energy prediction calculated from the momentum prediction and true particle type label"""
        return math.energy_from_momentum(self.momentum_prediction, self.true_labels)


class PositionPrediction(ABC):
    """
    Base class for position predictions.

    The following attributes are calculated and provided if the required truth information is given:

    ============================================================================================================
    `position_residuals`            (x,y,z) components of the true position minus reconstructed position
    `x_residuals`                   x component of the true position minus reconstructed position
    `y_residuals`                   y component of the true position minus reconstructed position
    `z_residuals`                   z component of the true position minus reconstructed position
    `position_3d_errors`            3D distance between the true and reconstructed position
    `position_longitudinal_errors`  True position minus reconstructed position in the particle's true direction
    `position_transverse_errors`    2D distance between the true and reconstructed position perpendicular to the
                                    particle's true direction
    ============================================================================================================
    """
    def __init__(self, true_positions=None, true_directions=None):
        """
        Create object to hold momentum predictions. If true positions are provided, the position residuals are
        calculated. If true directions are provided, also calculated is the decomposition of the position residuals into
        longitudinal (in direction of particle) and transverse (perpendicular to particle direction).

        Parameters
        ----------
        true_positions: array_like of float, optional
            Array of true positions, used in calculating residuals.
        true_directions: array_like of int, optional
            Array of true directions, used to decompose the position prediction residuals into longitudinal and
            transverse components.
        """
        self.true_positions = true_positions
        self.position_residuals = None
        self.x_residuals = None
        self.y_residuals = None
        self.z_residuals = None
        self.position_3d_errors = None
        self.position_longitudinal_errors = None
        self.position_transverse_errors = None
        if true_positions is not None:
            self.position_residuals = self.position_prediction - self.true_positions
            self.x_residuals = self.position_residuals[:, 0]
            self.y_residuals = self.position_residuals[:, 1]
            self.z_residuals = self.position_residuals[:, 2]
            if true_directions is not None:
                (self.position_3d_errors, self.position_longitudinal_errors,
                 self.position_transverse_errors) = math.decompose_along_direction(self.position_residuals, true_directions)
            else:
                self.position_3d_errors = np.linalg.norm(self.position_residuals, axis=-1)

    @property
    @abstractmethod
    def position_prediction(self):
        """This attribute gives the predicted positions"""


class DirectionPrediction(ABC):
    """
    Base class for direction predictions.

    The following attribute is calculated and provided if the required truth information is given:

    ===============================================================================================
    `direction_errors`              Angle between the true and reconstructed directions, in degrees
    ===============================================================================================
    """
    def __init__(self, true_directions=None):
        """
        Create object to hold momentum predictions.

        Parameters
        ----------
        true_directions: array_like of float, optional
            Array of true directions, used in calculating residuals
        """
        self.true_directions = true_directions
        self.direction_errors = None
        if true_directions is not None:
            self.direction_errors = math.angle_between_directions(self.direction_prediction, true_directions, degrees=True)

    @property
    @abstractmethod
    def direction_prediction(self):
        """This attribute gives the predicted directions"""


class FitQun1RingFit(RegressionRun, PositionPrediction, DirectionPrediction, MomentumPrediction):
    """Class to hold position, direction and momentum predictions of a fiTQun reconstruction run"""
    def __init__(self, fitqun_output, run_label, true_positions=None, true_directions=None, true_momenta=None,
                 true_labels=None, indices=..., selection=None, particle_label_map=None, **plot_args):
        """
        Construct object to hold position, direction and momentum predictions of a fiTQun reconstruction run.

        Parameters
        ----------
        fitqun_output: analysis.read.FiTQunOutput
            Output from a fiTQun reconstruction run
        run_label: str
            Label to describe this set of results to use in plot legends, etc.
        true_positions array_like of float, optional
            Array of true positions for the events in these reconstruction results
        true_directions array_like of int, optional
            Array of true directions for the events in these reconstruction results
        true_momenta array_like of int, optional
            Array of true momenta for the events in these reconstruction results
        true_labels: int or array_like of int, optional
            PID label or array of PID labels for the events in these reconstruction results
        indices: array_like of int, optional
            Array of indices of events to select out of the events in the fiTQun output (by default use all events).
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            (by default use all events).
        particle_label_map: dict
            Dictionary mapping particle type names to label integers. By default, use gamma:0, electron:1, muon:2, pi0:3
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results.
        """
        RegressionRun.__init__(self, run_label=run_label, selection=selection, **plot_args)
        self.fitqun_output = fitqun_output
        self.n_events = len(np.empty(len(fitqun_output.chain))[indices])
        if isinstance(true_labels, int):
            true_labels = np.repeat(true_labels, self.n_events)
        self.true_labels = true_labels
        if particle_label_map is None:
            particle_label_map = {'gamma': 0, 'electron': 1, 'muon': 2, 'pi0': 3}
        self.particle_label_map = particle_label_map
        self.label_set = set(true_labels)
        self.particle_indices = {p: (true_labels == l) for p, l in particle_label_map.items() if l in self.label_set}
        self._momentum_prediction = None
        self._position_prediction = None
        self._direction_prediction = None
        # use lambdas to delay prevent reading all the data until/unless it's actually used
        self.momentum_map = {
            'gamma': lambda: self.fitqun_output.electron_momentum[indices],
            'electron': lambda: self.fitqun_output.electron_momentum[indices],
            'muon': lambda: self.fitqun_output.muon_momentum[indices],
            'pi0': lambda: self.fitqun_output.pi0_momentum[indices],
        }
        self.position_map = {
            'gamma': lambda: self.fitqun_output.electron_position[indices],
            'electron': lambda: self.fitqun_output.electron_position[indices],
            'muon': lambda: self.fitqun_output.muon_position[indices],
            'pi0': lambda: self.fitqun_output.pi0_position[indices],
        }
        self.direction_map = {
            'gamma': lambda: self.fitqun_output.electron_direction[indices],
            'electron': lambda: self.fitqun_output.electron_direction[indices],
            'muon': lambda: self.fitqun_output.muon_direction[indices],
            'pi0': lambda: self.fitqun_output.pi0_direction[indices],
        }
        PositionPrediction.__init__(self, true_positions=true_positions, true_directions=true_directions)
        DirectionPrediction.__init__(self, true_directions=true_directions)
        MomentumPrediction.__init__(self, true_labels=true_labels, true_momenta=true_momenta)

    @property
    def momentum_prediction(self):
        """Momentum prediction from fiTQun for the single ring fit of the particle type(s) set during construction"""
        if self._momentum_prediction is None:
            self._momentum_prediction = np.zeros(self.n_events)
            for p, i in self.particle_indices.items():
                self._momentum_prediction[i] = self.momentum_map[p]()
        return self._momentum_prediction

    @property
    def position_prediction(self):
        """Position prediction from fiTQun for the single ring fit of the particle type(s) set during construction"""
        if self._position_prediction is None:
            self._position_prediction = np.zeros((self.n_events, 3))
            for p, i in self.particle_indices.items():
                self._position_prediction[i] = self.position_map[p]()
        return self._position_prediction

    @property
    def direction_prediction(self):
        """Direction prediction from fiTQun for the single ring fit of the particle type(s) set during construction"""
        if self._direction_prediction is None:
            self._direction_prediction = np.zeros((self.n_events, 3))
            for p, i in self.particle_indices.items():
                self._direction_prediction[i] = self.direction_map[p]()
        return self._direction_prediction


class WatChMaLRegression(RegressionRun, WatChMaLOutput, ABC):
    """Base class to hold predictions of a WatChMaL regression run"""
    def __init__(self, directory, run_label, indices=None, selection=None, **plot_args):
        """
        Constructs the object holding the results of a WatChMaL regression run.

        Parameters
        ----------
        directory: str
            Top-level output directory of a WatChMaL regression run.
        run_label: str
            Label to describe this set of results to use in plot legends, etc.
        indices: array_like of int, optional
            Array of indices of events to select out of the indices output by WatChMaL (by default use all events sorted
            by their indices).
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            By default, use all results.
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results.
        """
        RegressionRun.__init__(self, run_label=run_label, selection=selection, **plot_args)
        WatChMaLOutput.__init__(self, directory=directory, indices=indices)
        self._predictions = None

    def read_training_log_from_csv(self, directory):
        """
        Read the training progression logs from the given directory.

        Parameters
        ----------
        directory: str
            Path to the directory of the training run.

        Returns
        -------
        np.ndarray
            Array of train epoch values for each entry in the training progression log.
        np.ndarray
            Array of train loss values for each entry in the training progression log.
        np.ndarray
            Array of validation epoch values for each entry in the training progression log
        np.ndarray
            Array of validation loss values for each entry in the training progression log
        np.ndarray
            Array of boolean values indicating whether each entry had the best validation loss so far in the training
            progression log
        """
        super().read_training_log_from_csv(directory)
        self._val_log_loss = self._log_val[:, 2]
        self._val_log_best = self._log_val[:, 3].astype(bool)
        return self._train_log_epoch, self._train_log_loss, self._val_log_epoch, self._val_log_loss, self._val_log_best

    @property
    def predictions(self):
        """Predictions output from the WatChMaL regression run"""
        if self._predictions is None:
            self._predictions = self.get_outputs("predictions")
        return self._predictions


class WatChMaLPositionRegression(WatChMaLRegression, PositionPrediction):
    """Class to hold position predictions of a WatChMaL regression run"""
    def __init__(self, directory, run_label, true_positions=None, true_directions=None, indices=None, selection=None,
                 **plot_args):
        """
        Constructs the object holding the results of a WatChMaL position regression run.

        Parameters
        ----------
        directory: str
            Top-level output directory of a WatChMaL regression run.
        run_label: str
            Label to describe this set of results to use in plot legends, etc.
        true_positions: array_like of int, optional
            Array of true positions for the events in these regression results, to calculate position residuals
        true_directions: array_like of int, optional
            Array of true directions for the events in these regression results, to decompose position residuals
        indices: array_like of int, optional
            Array of indices of events to select out of the indices output by WatChMaL (by default use all events sorted
            by their indices).
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            By default, use all results.
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results.
        """
        WatChMaLRegression.__init__(self, directory=directory, run_label=run_label, indices=indices,
                                    selection=selection, **plot_args)
        PositionPrediction.__init__(self, true_positions=true_positions, true_directions=true_directions)

    @property
    def position_prediction(self):
        """Position predictions output from the WatChMaL regression run"""
        return self.predictions


class WatChMaLDirectionRegression(WatChMaLRegression, DirectionPrediction):
    """Class to hold direction predictions of a WatChMaL regression run"""
    def __init__(self, directory, run_label, true_directions=None, indices=None, zenith_axis=None, selection=None,
                 **plot_args):
        """
        Constructs the object holding the results of a WatChMaL direction regression run.

        Parameters
        ----------
        directory: str
            Top-level output directory of a WatChMaL regression run.
        run_label: str
            Label to describe this set of results to use in plot legends, etc.
        true_directions: array_like of int, optional
            Array of true directions for the events in these regression results to calculate direction prediction errors
        indices: array_like of int, optional
            Array of indices of events to select out of the indices output by WatChMaL (by default use all events sorted
            by their indices).
        zenith_axis: int, optional
            Axis representing the zenith direction, for converting from WatChMaL's predicted zenith and azimuth angles
            into unit vector directions. By default, use the default tank axis set in `analysis.utils.math`.
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            By default, use all results.
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results.
        """
        WatChMaLRegression.__init__(self, directory=directory, run_label=run_label, indices=indices,
                                    selection=selection, **plot_args)
        self._direction_prediction = None
        self.zenith_axis = zenith_axis
        DirectionPrediction.__init__(self, true_directions=true_directions)

    @property
    def direction_prediction(self):
        """Direction predictions calculated from predicted angles output from the WatChMaL regression run"""
        if self._direction_prediction is None:
            self._direction_prediction = math.direction_from_angles(self.predictions, self.zenith_axis)
        return self._direction_prediction


class WatChMaLEnergyRegression(WatChMaLRegression, MomentumPrediction):
    """Class to hold momentum predictions of a WatChMaL regression run"""
    def __init__(self, directory, run_label, true_momenta=None, true_labels=None, indices=None, selection=None,
                 **plot_args):
        """
        Constructs the object holding the results of a WatChMaL energy regression run.

        Parameters
        ----------
        directory: str
            Top-level output directory of a WatChMaL regression run.
        run_label: str
            Label to describe this set of results to use in plot legends, etc.
        true_momenta: array_like of float, optional
            Array of true momenta for the events in these regression results to calculate momentum prediction residuals
        true_labels: int or array_like of int, optional
            PID label or array of PID labels for the events in these reconstruction results
        indices: array_like of int, optional
            Array of indices of events to select out of the indices output by WatChMaL (by default use all events sorted
            by their indices).
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            By default, use all results.
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results.
        """
        WatChMaLRegression.__init__(self, directory=directory, run_label=run_label, indices=indices,
                                    selection=selection, **plot_args)
        self._momentum_prediction = None
        MomentumPrediction.__init__(self, true_momenta=true_momenta, true_labels=true_labels)

    @property
    def momentum_prediction(self):
        """Momentum predictions calculated from predicted energies output from the WatChMaL regression run"""
        if self._momentum_prediction is None:
            self._momentum_prediction = math.momentum_from_energy(self.predictions, self.true_labels)
        return self._momentum_prediction


class CombinedRegressionRun(RegressionRun):
    """
    Class used to combine different regression runs that each provide different predicted quantities. The combined run
    can be used like a single run that provides the predictions of each of the combined runs. When getting an attribute
    such as predictions or residuals, each of the combined runs is searched in turn until a run with the desired
    attribute is provided.
    """
    def __init__(self, combined_runs, run_label=None, selection=None, **plot_args):
        """

        Parameters
        ----------
        combined_runs: sequence of RegressionRun
            Set of regression runs providing different predictions to combine.
        run_label: str, optional
            Label to describe this set of results to use in plot legends, etc. By default, use the label of the first of
            the combined runs.
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            By default, use the selection of the first of the combined runs.
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results. By default, use the plot arguments of the first of the combined runs.
        """
        self.runs = combined_runs
        first_run = next(iter(combined_runs))
        if run_label is None:
            run_label = first_run.run_label
        if not plot_args:
            plot_args = first_run.plot_args
        if selection is None:
            selection = first_run.selection
        RegressionRun.__init__(self, run_label=run_label, selection=selection, **plot_args)

    def __getattr__(self, attr):
        # Loop over the combined runs and look for the attribute in each, or raise exception if it's not found in any.
        for r in self.runs:
            if hasattr(r, attr):
                return getattr(r, attr)
        raise AttributeError(attr)
