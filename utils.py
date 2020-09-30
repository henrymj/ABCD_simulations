import numpy as np
import pandas as pd


class SimulateData():

    def __init__(self, model='independent_race',
                 variable_mu_stop=False,
                 trigger_failures=False,
                 guesses=False,
                 mu_go_grader=None):
        self.model = model
        self.variable_mu_stop = variable_mu_stop
        self.trigger_failures = trigger_failures
        self.guesses = guesses
        trial_iterators = {
            'independent_race': self._independent_race_trial,
            'interactive_race': self._interactive_race_trial,
            'blocked_input': self._blocked_input_trial
        }
        self._trial_iter = trial_iterators[model]

        self._mu_go_grader = None
        if mu_go_grader:
            mu_go_graders = {
                'log': self._log_mu_go,
                'linear': self._linear_mu_go,
            }
            assert mu_go_grader in ['log', 'linear']
            self._mu_go_grader = mu_go_graders[mu_go_grader]

    def simulate(self, params={}):
        params = self._init_params(params)
        data_dict = self._init_data_dict()
        self._set_n_trials(params)
        self._set_n_guesses(params)
        for ssd_idx, SSD in enumerate(params['SSDs']):
            data_dict = self._simulate_guesses(data_dict, params, SSD)
            data_dict = self._simulate_stop_trials(data_dict, params,
                                                   SSD)
        data_dict = self._simulate_go_trials(data_dict, params)

        # convert to dataframe
        data_df = pd.DataFrame.from_dict(data_dict)
        data_df['block'] = 0
        for rt_type in ['go', 'stop']:
            data_df['{}RT'.format(rt_type)] = np.where(
                data_df['condition'] == rt_type,
                data_df['RT'],
                np.nan)
        del data_df['RT']

        return data_df

    def _simulate_guesses(self, data_dict, params, SSD):
        if SSD is None:  # go trials
            n_guess = int(self._n_guess_go)
        else:
            n_guess = int(self._n_guess_stop[SSD])
        if n_guess == 0:
            return data_dict
        guess_RTs = params['guess_function'](
            n_guess
        )
        stop_init_time = SSD + params['nondecision_stop'] if\
            (SSD is not None) else np.nan
        for trial_idx, guess_RT in enumerate(guess_RTs):
            trial = self._init_trial_dict(params, trial_idx,
                                          SSD=SSD,
                                          stop_init_time=stop_init_time)
            if SSD is not None:
                stop_accum = 0
                for time in range(1, trial['max_time']+1):
                    if time >= trial['stop_init_time']:
                        stop_accum = self._at_least_0(
                            stop_accum + trial['mu_stop'] +
                            np.random.normal(loc=0, scale=trial['noise_stop'])
                        )
                        trial['process_stop'].append(stop_accum)
                    if stop_accum > trial['threshold']:
                        break

                if guess_RT <= time:
                    trial['RT'] = guess_RT
            else:
                if guess_RT <= trial['max_time']:
                    trial['RT'] = guess_RT
            data_dict = self._update_data_dict(data_dict, trial)
        return data_dict

    def _simulate_go_trials(self, data_dict, params):
        data_dict = self._simulate_guesses(data_dict, params, None)
        for trial_idx in range(int(self._n_guess_go),
                               self._n_trials_go):
            trial = self._init_trial_dict(params, trial_idx, condition='go')
            go_accum = 0
            stop_accum = 0
            for time in range(1, trial['max_time']+1):
                if time >= trial['nondecision_go']:
                    go_accum = self._at_least_0(
                        go_accum + trial['mu_go'] +
                        np.random.normal(loc=0, scale=trial['noise_go'])
                    )
                    trial['process_go'].append(go_accum)
                if go_accum > trial['threshold']:
                    trial['RT'] = time
                    break

            trial['accum_go'] = go_accum
            trial['accum_stop'] = stop_accum
            data_dict = self._update_data_dict(data_dict, trial)
        return data_dict

    def _simulate_stop_trials(self, data_dict, params, SSD):
        stop_init_time = SSD + params['nondecision_stop']
        for trial_idx in range(int(self._n_guess_stop[SSD]),
                               int(self._n_trials_stop[SSD])):
            trial = self._init_trial_dict(params, trial_idx,
                                          SSD=SSD,
                                          stop_init_time=stop_init_time)
            data_dict = self._trial_iter(data_dict, trial)
        return data_dict

    def _independent_race_trial(self, data_dict, trial):
        go_accum = 0
        stop_accum = 0
        for time in range(1, trial['max_time']+1):
            if time >= trial['stop_init_time']:
                stop_accum = self._at_least_0(
                    stop_accum + trial['mu_stop'] +
                    np.random.normal(loc=0, scale=trial['noise_stop'])
                )
                trial['process_stop'].append(stop_accum)
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
                trial['process_go'].append(go_accum)
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break
            if stop_accum > trial['threshold']:
                break

        trial['accum_go'] = go_accum
        trial['accum_stop'] = stop_accum
        return self._update_data_dict(data_dict, trial)

    def _interactive_race_trial(self, data_dict, trial):
        go_accum = 0
        stop_accum = 0
        for time in range(1, trial['max_time']+1):
            if time >= trial['stop_init_time']:
                stop_accum = self._at_least_0(
                    stop_accum + trial['mu_stop'] +
                    np.random.normal(loc=0, scale=trial['noise_stop'])
                )
                trial['process_stop'].append(stop_accum)
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] -
                    trial['inhibition_interaction']*stop_accum +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
                trial['process_go'].append(go_accum)
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break
        trial['accum_go'] = go_accum
        trial['accum_stop'] = stop_accum
        return self._update_data_dict(data_dict, trial)

    def _blocked_input_trial(self, data_dict, trial):
        go_accum = 0
        stop_accum = 0
        for time in range(1, trial['max_time']+1):
            if time >= trial['stop_init_time']:
                stop_accum = self._at_least_0(
                    trial['mu_stop'] + np.random.normal(
                        loc=0, scale=trial['noise_stop'])
                )
                trial['process_stop'].append(stop_accum)
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] -
                    trial['inhibition_interaction']*stop_accum +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
                trial['process_go'].append(go_accum)
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break

        trial['accum_go'] = go_accum
        trial['accum_stop'] = stop_accum
        return self._update_data_dict(data_dict, trial)

    def _set_n_trials(self, params):
        num_SSDs = len(params['SSDs'])
        n_trials_stop = params['n_trials_stop']
        if type(params['n_trials_stop']) in [float, int]:
            n_trials_stop = [params['n_trials_stop']] * num_SSDs
        elif type(params['n_trials_stop']) in [list, np.ndarray]:
            if len(params['n_trials_stop']) == 1:
                n_trials_stop = params['n_trials_stop'] * num_SSDs
            else:
                n_trials_stop = params['n_trials_stop']

        assert(len(n_trials_stop) == num_SSDs)

        self._n_trials_go = params['n_trials_go']
        self._n_trials_stop = {SSD: n for SSD, n in
                               zip(params['SSDs'], n_trials_stop)}

    def _set_n_guesses(self, params):
        # TODO: ADD ASSERTIONS TO CHECK FOR CORRECT USES, clean up!!!
        # TODO: allow for guessing on go trials
        num_SSDs = len(params['SSDs'])
        if self.guesses:
            p_guess_go = params['p_guess_go']
            if type(params['p_guess_stop']) in [float, int]:
                p_guess_per_SSD = [params['p_guess_stop']] * num_SSDs
            elif type(params['p_guess_stop']) in [list, np.ndarray]:
                if len(params['p_guess_stop']) == 1:
                    p_guess_per_SSD = params['p_guess_stop'] * num_SSDs
                else:
                    p_guess_per_SSD = params['p_guess_stop']
            else:
                print('did not expect type {}'.format(
                      type(params['p_guess_stop'])))
        else:
            p_guess_per_SSD = [0] * num_SSDs
            p_guess_go = 0
        assert(len(p_guess_per_SSD) == num_SSDs)

        # TODO: clean up these lines? -
        # if 0 is returned, it's viewed as an int,
        # not a float, so it needs to be converted
        self._n_guess_go = self._at_least_0(
            np.rint(float(
                    p_guess_go * self._n_trials_go)))
        self._n_guess_stop = {SSD: self._at_least_0(
                                np.rint(float(p * self._n_trials_stop[SSD])))
                              for SSD, p in zip(params['SSDs'],
                                                p_guess_per_SSD)}

    def _get_mu_stop(self, params):
        mu_stop = params['mu_stop']
        if self.variable_mu_stop:
            mu_stop = mu_stop+np.random.normal(
                loc=0, scale=params['noise_stop']*.7)
        if self.trigger_failures and np.random.uniform(0, 1) <\
                params['p_trigger_fail']:
            mu_stop = 0
        return self._at_least_0(mu_stop)

    def _get_mu_go(self, params, SSD):
        # TODO: make more dynamic, pass max_SSD
        mu_go = params['mu_go']
        if self._mu_go_grader and SSD is not None:
            mu_go = self._mu_go_grader(mu_go, SSD)
        return mu_go

    def _log_mu_go(self, mu_go, SSD, max_SSD=550):
        if SSD > max_SSD:
            SSD = max_SSD
        return self._at_least_0((np.log(SSD/max_SSD)/4+1) * mu_go)

    def _linear_mu_go(self, mu_go, SSD, max_SSD=550):
        if SSD > max_SSD:
            SSD = max_SSD
        return self._at_least_0((SSD/max_SSD) * mu_go)

    def _init_params(self, params):
        # TODO: move default dict to json, read in
        default_dict = {'mu_go': .2,
                        'mu_stop': .4,
                        'noise_go':  3.5,  # 2,  # 1.13,
                        'noise_stop': 3,  # 2,  # 1.75,
                        'threshold': 100,
                        'nondecision_go': 50,
                        'nondecision_stop': 50,
                        'inhibition_interaction': .5,
                        'SSDs': np.arange(0, 600, 50),
                        'n_trials_go': 1000,
                        'n_trials_stop': 1000,
                        'max_time': 3000,
                        'p_trigger_fail': 0,
                        'p_guess_go': 0,
                        'p_guess_stop': 0,
                        'guess_function': lambda x: np.random.uniform(
                            200, 400, x),
                        'mu_go_grader': 'log'
                        }

        for key in default_dict:
            if key not in params:
                params[key] = default_dict[key]        

        return params

    def _init_data_dict(self):
        return {
            'condition': [],
            'SSD': [],
            'trial_idx': [],
            'RT': [],
            'mu_go': [],
            'mu_stop': [],
            'accum_go': [],
            'accum_stop': [],
            'process_go': [],
            'process_stop': [],
            }

    def _init_trial_dict(self, params, trial_idx,
                         SSD=None, stop_init_time=np.nan, condition='stop'):
        trial = {
                'condition': condition,
                'SSD': SSD,
                'trial_idx': trial_idx,
                'mu_go': self._get_mu_go(params, SSD),
                'mu_stop': self._get_mu_stop(params),
                'stop_init_time': stop_init_time,
                'noise_go': params['noise_go'],
                'noise_stop': params['noise_stop'],
                'nondecision_go': params['nondecision_go'],
                'inhibition_interaction': params['inhibition_interaction'],
                'threshold': params['threshold'],
                'max_time': params['max_time'],
                'accum_go': np.nan,
                'accum_stop': np.nan,
                'process_go': [],
                'process_stop': [],
                'RT': np.nan
            }
        return trial

    def _update_data_dict(self, data_dict, update_dict):
        for key in data_dict.keys():
            data_dict[key].append(update_dict[key])
        return data_dict

    def _at_least_0(self, num):
        if num < 0:
            num = 0
        return num


# EDITED FROM https://github.com/sbebo/joypy/blob/master/joypy/joyplot.py
import os
from scipy.stats import gaussian_kde
import warnings

try:
    from pandas.plotting._tools import (_subplots, _flatten)
except:
    # TODO this is a quick fix for #38
    from pandas.plotting._matplotlib.tools import (_subplots, _flatten)

from pandas import (DataFrame, Series)
from pandas.core.dtypes.common import is_number
from pandas.core.groupby import DataFrameGroupBy
from matplotlib import pyplot as plt
from warnings import warn

_DEBUG = False


def _x_range(data, extra=0.2):
    """ Compute the x_range, i.e., the values for which the
        density will be computed. It should be slightly larger than
        the max and min so that the plot actually reaches 0, and
        also has a bit of a tail on both sides.
    """
    try:
        sample_range = np.nanmax(data) - np.nanmin(data)
    except ValueError:
        return []
    if sample_range < 1e-6:
        return [np.nanmin(data), np.nanmax(data)]
    return np.linspace(np.nanmin(data) - extra*sample_range,
                       np.nanmax(data) + extra*sample_range, 1000)

def _setup_axis(ax, x_range, col_name=None, grid=False, ylabelsize=None, yrot=None):
    """ Setup the axis for the joyplot:
        - add the y label if required (as an ytick)
        - add y grid if required
        - make the background transparent
        - set the xlim according to the x_range
        - hide the xaxis and the spines
    """
    if col_name is not None:
        ax.set_yticks([0])
        ax.set_yticklabels([col_name], fontsize=ylabelsize, rotation=yrot)
        ax.yaxis.grid(grid)
    else:
        ax.yaxis.set_visible(False)
    ax.patch.set_alpha(0)
    ax.set_xlim([min(x_range), max(x_range)])
    ax.tick_params(axis='both', which='both', length=0, pad=10)
    ax.xaxis.set_visible(_DEBUG)
    ax.set_frame_on(_DEBUG)

def _is_numeric(x):
    """ Whether the array x is numeric. """
    return all(is_number(i) for i in x)

def _get_alpha(i, n, start=0.4, end=1.0):
    """ Compute alpha value at position i out of n """
    return start + (1 + i)*(end - start)/n

def _remove_na(l):
    """ Remove NA values. Should work for lists, arrays, series. """
    return Series(l).dropna().values

def _moving_average(a, n=3, zero_padded=False):
    """ Moving average of order n.
        If zero padded, returns an array of the same size as
        the input: the values before a[0] are considered to be 0.
        Otherwise, returns an array of length len(a) - n + 1 """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    if zero_padded:
        return ret / n
    else:
        return ret[n - 1:] / n

def joyplot(data, column=None, by=None, grid=False,
            xlabelsize=None, xrot=None, ylabelsize=None, yrot=None,
            ax=None, figsize=None,
            hist=False, bins=10,
            fade=False, ylim='max',
            fill=True, linecolor=None,
            overlap=1, background=None,
            labels=None, xlabels=True, ylabels=True,
            range_style='all',
            x_range=None,
            title=None,
            colormap=None,
            color=None,
            **kwds):
    """
    Draw joyplot of a DataFrame, or appropriately nested collection,
    using matplotlib and pandas.
    A joyplot is a stack of vertically aligned density plots / histograms.
    By default, if 'data' is a DataFrame,
    this function will plot a density plot for each column.
    This wrapper method tries to convert whatever structure is given
    to a nested collection of lists with additional information
    on labels, and use the private _joyplot function to actually
    draw theh plot.
    Parameters
    ----------
    data : DataFrame, Series or nested collection
    column : string or sequence
        If passed, will be used to limit data to a subset of columns
    by : object, optional
        If passed, used to form separate plot groups
    grid : boolean, default True
        Whether to show axis grid lines
    labels : boolean or list, default True.
        If list, must be the same size of the de
    xlabelsize : int, default None
        If specified changes the x-axis label size
    xrot : float, default None
        rotation of x axis labels
    ylabelsize : int, default None
        If specified changes the y-axis label size
    yrot : float, default None
        rotation of y axis labels
    ax : matplotlib axes object, default None
    figsize : tuple
        The size of the figure to create in inches by default
    hist : boolean, default False
    bins : integer, default 10
        Number of histogram bins to be used
    color : color or colors to be used in the plots. It can be:
        a string or anything interpretable as color by matplotib;
        a list of colors. See docs / examples for more details.
    kwds : other plotting keyword arguments
        To be passed to hist/kde plot function
    """

    if column is not None:
        if not isinstance(column, (list, np.ndarray)):
            column = [column]

    def _grouped_df_to_standard(grouped, column):
        converted = []
        labels = []
        for i, (key, group) in enumerate(grouped):
            if column is not None:
                group = group[column]
            labels.append(key)
            converted.append([_remove_na(group[c]) for c in group.columns if _is_numeric(group[c])])
            if i == 0:
                sublabels = [col for col in group.columns if _is_numeric(group[col])]
        return converted, labels, sublabels

    #################################################################
    # GROUPED
    # - given a grouped DataFrame, a group by key, or a dict of dicts of Series/lists/arrays
    # - select the required columns/Series/lists/arrays
    # - convert to standard format: list of lists of non-null arrays
    #   + extra parameters (labels and sublabels)
    #################################################################
    if isinstance(data, DataFrameGroupBy):
        grouped = data
        converted, _labels, sublabels = _grouped_df_to_standard(grouped, column)
        if labels is None:
            labels = _labels
    elif by is not None and isinstance(data, DataFrame):
        grouped = data.groupby(by)
        if column is None:
            # Remove the groupby key. It's not automatically removed by pandas.
            column = list(data.columns).remove(by)
        converted, _labels, sublabels = _grouped_df_to_standard(grouped, column)
        if labels is None:
            labels = _labels
        # If there is at least an element which is not a list of lists.. go on.
    elif isinstance(data, dict) and all(isinstance(g, dict) for g in data.values()):
        grouped = data
        if labels is None:
            labels = list(grouped.keys())
        converted = []
        for i, (key, group) in enumerate(grouped.items()):
            if column is not None:
                converted.append([_remove_na(g) for k,g in group.items() if _is_numeric(g) and k in column])
                if i == 0:
                    sublabels = [k for k,g in group.items() if _is_numeric(g)]
            else:
                converted.append([_remove_na(g) for k,g in group.items() if _is_numeric(g)])
                if i == 0:
                    sublabels = [k for k,g in group.items() if _is_numeric(g)]
    #################################################################
    # PLAIN:
    # - given a DataFrame or list/dict of Series/lists/arrays
    # - select the required columns/Series/lists/arrays
    # - convert to standard format: list of lists of non-null arrays + extra parameter (labels)
    #################################################################
    elif isinstance(data, DataFrame):
        if column is not None:
            data = data[column]
        converted = [[_remove_na(data[col])] for col in data.columns if _is_numeric(data[col])]
        labels = [col for col in data.columns if _is_numeric(data[col])]
        sublabels = None
    elif isinstance(data, dict):
        if column is not None:
            converted = [[_remove_na(g)] for k,g in data.items() if _is_numeric(g) and k in column]
            labels = [k for k,g in data.items() if _is_numeric(g) and k in column]
        else:
            converted = [[_remove_na(g)] for k,g in data.items() if _is_numeric(g)]
            labels = [k for k,g in data.items() if _is_numeric(g)]
        sublabels = None
    elif isinstance(data, list):
        if column is not None:
            converted = [[_remove_na(g)] for g in data if _is_numeric(g) and i in column]
        else:
            converted = [[_remove_na(g)] for g in data if _is_numeric(g)]
        if labels and len(labels) != len(converted):
            raise ValueError("The number of labels does not match the length of the list.")

        sublabels = None
    else:
        raise TypeError("Unknown type for 'data': {!r}".format(type(data)))

    if ylabels is False:
        labels = None

    if all(len(subg)==0 for g in converted for subg in g):
        raise ValueError("No numeric values found. Joyplot requires at least a numeric column/group.")

    if any(len(subg)==0 for g in converted for subg in g):
        warn("At least a column/group has no numeric values.")


    return _joyplot(converted, labels=labels, sublabels=sublabels,
                    grid=grid,
                    xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot,
                    ax=ax, figsize=figsize,
                    hist=hist, bins=bins,
                    fade=fade, ylim=ylim,
                    fill=fill, linecolor=linecolor,
                    overlap=overlap, background=background,
                    xlabels=xlabels,
                    range_style=range_style, x_range=x_range,
                    title=title,
                    colormap=colormap,
                    color=color,
                    **kwds)

###########################################

def plot_density(ax, x_range, v, kind="kde", bw_method=None,
                 bins=50,
                 fill=False, linecolor=None, clip_on=True, **kwargs):
    """ Draw a density plot given an axis, an array of values v and an array
        of x positions where to return the estimated density.
    """
    v = _remove_na(v)
    if len(v) == 0 or len(x_range) == 0:
        return

    if kind == "kde":
        try:
            gkde = gaussian_kde(v, bw_method=bw_method)
            y = gkde.evaluate(x_range)
        except ValueError:
            # Handle cases where there is no data in a group.
            y = np.zeros_like(x_range)
        except np.linalg.LinAlgError as e:
            # Handle singular matrix in kde computation.
            distinct_values = np.unique(v)
            if len(distinct_values) == 1:
                # In case of a group with a single value val,
                # that should have infinite density,
                # return a δ(val)
                val = distinct_values[0]
                warnings.warn("The data contains a group with a single distinct value ({}) "
                              "having infinite probability density. "
                              "Consider using a different visualization.".format(val))

                # Find index i of x_range
                # such that x_range[i-1] < val ≤ x_range[i]
                i = np.searchsorted(x_range, val)

                y = np.zeros_like(x_range)
                y[i] = 1
            else:
                raise e

    elif kind == "counts":
        y, bin_edges = np.histogram(v, bins=bins, range=(min(x_range), max(x_range)))
        # np.histogram returns the edges of the bins.
        # We compute here the middle of the bins.
        x_range = _moving_average(bin_edges, 2)
    elif kind == "normalized_counts":
        y, bin_edges = np.histogram(v, bins=bins, density=False,
                                    range=(min(x_range), max(x_range)))
        # np.histogram returns the edges of the bins.
        # We compute here the middle of the bins.
        y = y / len(v)
        x_range = _moving_average(bin_edges, 2)
    elif kind == "values":
        # Warning: to use values and get a meaningful visualization,
        # x_range must also be manually set in the main function.
        y = v
        x_range = list(range(len(y)))
    else:
        raise NotImplementedError

    if fill:
        ax.fill_between(x_range, 0.0, y, clip_on=clip_on, **kwargs)

        # Hack to have a border at the bottom at the fill patch
        # (of the same color of the fill patch)
        # so that the fill reaches the same bottom margin as the edge lines
        # with y value = 0.0
        kw = kwargs
        kw["label"] = None
        ax.plot(x_range, [0.0]*len(x_range), clip_on=clip_on, **kw)

    if linecolor is not None:
        kwargs["color"] = linecolor

    # Remove the legend labels if we are plotting filled curve:
    # we only want one entry per group in the legend (if shown).
    if fill:
        kwargs["label"] = None

    ax.plot(x_range, y, clip_on=clip_on, **kwargs)

###########################################

def _joyplot(data,
             grid=False,
             labels=None, sublabels=None,
             xlabels=True,
             xlabelsize=None, xrot=None,
             ylabelsize=None, yrot=None,
             ax=None, figsize=None,
             hist=False, bins=10,
             fade=False,
             xlim=None, ylim='max',
             fill=True, linecolor=None,
             overlap=1, background=None,
             range_style='all', x_range=None, tails=0.2,
             title=None,
             legend=False, loc="upper right",
             colormap=None, color=None,
             **kwargs):
    """
    Internal method.
    Draw a joyplot from an appropriately nested collection of lists
    using matplotlib and pandas.
    Parameters
    ----------
    data : DataFrame, Series or nested collection
    grid : boolean, default True
        Whether to show axis grid lines
    labels : boolean or list, default True.
        If list, must be the same size of the de
    xlabelsize : int, default None
        If specified changes the x-axis label size
    xrot : float, default None
        rotation of x axis labels
    ylabelsize : int, default None
        If specified changes the y-axis label size
    yrot : float, default None
        rotation of y axis labels
    ax : matplotlib axes object, default None
    figsize : tuple
        The size of the figure to create in inches by default
    hist : boolean, default False
    bins : integer, default 10
        Number of histogram bins to be used
    kwarg : other plotting keyword arguments
        To be passed to hist/kde plot function
    """
    if fill is True and linecolor is None:
        linecolor = "k"

    if sublabels is None:
        legend = False

    def _get_color(i, num_axes, j, num_subgroups):
        if isinstance(color, list):
            return color[j] if num_subgroups > 1 else color[i]
        elif color is not None:
            return color
        elif isinstance(colormap, list):
            return colormap[j](i/num_axes)
        elif color is None and colormap is None:
            num_cycle_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            return plt.rcParams['axes.prop_cycle'].by_key()['color'][j % num_cycle_colors]
        else:
            return colormap(i/num_axes)

    ygrid = (grid is True or grid == 'y' or grid == 'both')
    xgrid = (grid is True or grid == 'x' or grid == 'both')

    num_axes = len(data)

    if x_range is None:
        global_x_range = _x_range([v for g in data for sg in g for v in sg])
        global_x_range = [i for i in global_x_range if i >= 0 and i <=3000]
    else:
        global_x_range = _x_range(x_range, 0.0)
    global_x_min, global_x_max = min(global_x_range), max(global_x_range)

    # Each plot will have its own axis
    fig, axes = _subplots(naxes=num_axes, ax=ax, squeeze=False,
                          sharex=True, sharey=False, figsize=figsize,
                          layout_type='vertical')
    _axes = _flatten(axes)

    # The legend must be drawn in the last axis if we want it at the bottom.
    if loc in (3, 4, 8) or 'lower' in str(loc):
        legend_axis = num_axis - 1
    else:
        legend_axis = 0

    # A couple of simple checks.
    if labels is not None:
        assert len(labels) == num_axes
    if sublabels is not None:
        assert all(len(g) == len(sublabels) for g in data)
    if isinstance(color, list):
        assert all(len(g) <= len(color) for g in data)
    if isinstance(colormap, list):
        assert all(len(g) == len(colormap) for g in data)

    for i, group in enumerate(data):
        a = _axes[i]
        group_zorder = i
        if fade:
            kwargs['alpha'] = _get_alpha(i, num_axes)

        num_subgroups = len(group)
        
        if hist:
            # matplotlib hist() already handles multiple subgroups in a histogram
            a.hist(group, label=sublabels, bins=bins,
                  histtype=u'step', edgecolor=linecolor, **kwargs)
            a.hist(group, label=sublabels, bins=bins,
                   color= colormap(i/num_axes),
                   range=[min(global_x_range), max(global_x_range)],
                   zorder=group_zorder, **kwargs)
        else:
            for j, subgroup in enumerate(group):

                # Compute the x_range of the current plot
                if range_style == 'all':
                # All plots have the same range
                    x_range = global_x_range
                elif range_style == 'own':
                # Each plot has its own range
                    x_range = _x_range(subgroup, tails)
                elif range_style == 'group':
                # Each plot has a range that covers the whole group
                    x_range = _x_range(group, tails)
                elif isinstance(range_style, (list, np.ndarray)):
                # All plots have exactly the range passed as argument
                    x_range = _x_range(range_style, 0.0)
                else:
                    raise NotImplementedError("Unrecognized range style.")

                if sublabels is None:
                    sublabel = None
                else:
                    sublabel = sublabels[j]

                element_zorder = group_zorder + j/(num_subgroups+1)
                element_color = _get_color(i, num_axes, j, num_subgroups)

                plot_density(a, x_range, subgroup,
                             fill=fill, linecolor=linecolor, label=sublabel,
                             zorder=element_zorder, color=element_color,
                             bins=bins, **kwargs)


        # Setup the current axis: transparency, labels, spines.
        col_name = None if labels is None else labels[i]
        _setup_axis(a, global_x_range, col_name=col_name, grid=ygrid,
                ylabelsize=ylabelsize, yrot=yrot)

        # When needed, draw the legend
        if legend and i == legend_axis:
            a.legend(loc=loc)
            # Bypass alpha values, in case
            for p in a.get_legend().get_patches():
                p.set_facecolor(p.get_facecolor())
                p.set_alpha(1.0)
            for l in a.get_legend().get_lines():
                l.set_alpha(1.0)


    # Final adjustments

    # Set the y limit for the density plots.
    # Since the y range in the subplots can vary significantly,
    # different options are available.
    if ylim == 'max':
        # Set all yaxis limit to the same value (max range among all)
        max_ylim = max(a.get_ylim()[1] for a in _axes)
        min_ylim = min(a.get_ylim()[0] for a in _axes)
        for a in _axes:
            a.set_ylim([min_ylim - 0.1*(max_ylim-min_ylim), max_ylim])

    elif ylim == 'own':
        # Do nothing, each axis keeps its own ylim
        pass

    else:
        # Set all yaxis lim to the argument value ylim
        try:
            for a in _axes:
                a.set_ylim(ylim)
        except:
            print("Warning: the value of ylim must be either 'max', 'own', or a tuple of length 2. The value you provided has no effect.")

    # Compute a final axis, used to apply global settings
    last_axis = fig.add_subplot(1, 1, 1)

    # Background color
    if background is not None:
        last_axis.patch.set_facecolor(background)

    for side in ['top', 'bottom', 'left', 'right']:
        last_axis.spines[side].set_visible(_DEBUG)

    # This looks hacky, but all the axes share the x-axis,
    # so they have the same lims and ticks
    last_axis.set_xlim(_axes[0].get_xlim())
    if xlabels is True:
        last_axis.set_xticks(np.array(_axes[0].get_xticks()[1:-1]))
        for t in last_axis.get_xticklabels():
            t.set_visible(True)
            t.set_fontsize(xlabelsize)
            t.set_rotation(xrot)

        # If grid is enabled, do not allow xticks (they are ugly)
        if xgrid:
            last_axis.tick_params(axis='both', which='both',length=0)
    else:
        last_axis.xaxis.set_visible(False)

    last_axis.yaxis.set_visible(False)
    last_axis.grid(xgrid)


    # Last axis on the back
    last_axis.zorder = min(a.zorder for a in _axes) - 1
    _axes = list(_axes) + [last_axis]

    if title is not None:
        plt.title(title)


    # The magic overlap happens here.
    h_pad = 5 + (- 5*(1 + overlap))
    fig.tight_layout(h_pad=h_pad)


    return fig, _axes