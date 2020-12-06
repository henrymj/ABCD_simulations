# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Likelihood-free estimation of stop model parameters
#
# This notebook shows how to use approximate Bayesian computation via sequential Monte Carlo (ABC-SMC) using the pyABC package.

# %%
# imports

import pyabc
import json
from pyabc import (ABCSMC,
                   RV, Distribution)
import numpy as np
import scipy.stats as st
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging

from ssd import fixedSSD
from stoptaskstudy import StopTaskStudy
# %matplotlib inline



# %%
# set to True for debugging outputs

debug = True

if debug:
    df_logger = logging.getLogger('Distance')
    df_logger.setLevel(logging.DEBUG)



# %%
# create a single-layer metrics dict for output
# since the pickler can't handle multilevel dicts
def cleanup_metrics(metrics):
    for k in metrics['SSRT']:
        metrics['SSRT_' + k] = metrics['SSRT'][k]
    del metrics['SSRT']
    return(metrics)


def stopsignal_model(parameters):
    paramfile = 'params.json'
    with open(paramfile) as f:
            params = json.load(f)
    # install the parameters from the simulation
    parameters['nondecision'] = int(parameters['nondecision'])
    params['mu']['go'] = parameters['mu_go']
    params['mu']['stop'] = parameters['mu_go'] + parameters['mu_stop_delta']
    params['mu_delta_incorrect']  = parameters['mu_delta_incorrect']
    params['noise_sd'] = {'go': parameters['noise_sd'],
                          'stop': parameters['noise_sd']}
    params['nondecision'] = {'go': parameters['nondecision'],
                             'stop': parameters['nondecision']}
    #print(params)
    # TBD
    #    if args.p_guess_file is not None:
    #        p_guess = pd.read_csv(args.p_guess_file, index_col=0)
    #        assert 'SSD' in p_guess.columns and 'p_guess' in p_guess.columns

    #    if args.random_seed is not None:
    #        np.random.seed(args.random_seed)

    min_ssd, max_ssd, ssd_step = 0, 550, 50
    ssd = fixedSSD(np.arange(min_ssd, max_ssd + ssd_step, ssd_step))

    study = StopTaskStudy(ssd, None, params=params)

    trialdata = study.run()
    metrics = study.get_stopsignal_metrics()
    # summarize data - go trials are labeled with SSD of -inf so that
    # they get included in the summary
    stop_data = trialdata.groupby('SSD').mean().query('SSD >= 0').resp.values
    results = {}
    
    metrics = cleanup_metrics(metrics)
    for k in [ 'mean_go_RT', 'mean_stopfail_RT', 'go_acc']:
        results.update({k: metrics[k]})
    for i, value in enumerate(stop_data):
        results[f'presp_{i}'] = value

    return(results)

parameter_prior = Distribution(mu_go=RV("uniform", 0, 1),
                               mu_stop_delta=RV("uniform", 0, 1),
                              mu_delta_incorrect=RV("uniform", 0, 1),
                              noise_sd=RV("uniform", 1, 4),
                              nondecision=RV("uniform", 40, 100))
parameter_prior.get_parameter_names()


# %%
params={'mu_delta_incorrect': 0.10386248711279868,
 'mu_go': 0.11422675271126799,
 'mu_stop_delta': 0.7850423871897488,
 'noise_sd': 3.1238287051634597,
       'nondecision':50}
simulation = stopsignal_model(params)
simulation


# %%

def rmse(a, b):
    return(np.sqrt(np.sum((a - b)**2)))

# sum errors for presp, gort, and stopfailrt
# scaling factors were determined by hand to roughly equate the rmse
# for the different result variables
def distance(simulation, data):
    presp_rmse = rmse(simulation['presp_0'], data['presp_0'])*3
    gort_rmse = rmse(simulation['mean_go_RT'], data['mean_go_RT'])/10
    goacc_rmse = rmse(simulation['go_acc'], data['go_acc']) * 10
    stopfailrt_rmse = rmse(simulation['mean_stopfail_RT'], data['stopfail_rt'])/10
    return(presp_rmse + gort_rmse + stopfailrt_rmse + goacc_rmse)

def distance2(simulation, data):
    sum_rmse = 0
    for k in simulation:
        sum_rmse += rmse(simulation[k], data[k])
    return(sum_rmse)

distance_adaptive = pyabc.AdaptivePNormDistance(p=2)
distance_fixed = pyabc.PNormDistance(p=2)
abc = ABCSMC(stopsignal_model, parameter_prior, distance_adaptive)


# %%
db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))
observed_presp = pd.read_csv('presp_by_ssd_inperson.txt',  delimiter=r"\s+", index_col=0)

observed_data = {'mean_go_RT': 455.367, 'mean_stopfail_RT': 219.364, 'go_acc': .935}
for i, value in enumerate(observed_presp.presp.values):
    observed_data[f'presp_{i}'] = value

# "presp": observed_presp.presp.values,
abc.new(db_path, observed_data)

# %%
print(distance_adaptive(simulation, observed_data))
print(distance2(simulation, observed_data))

# %%
history = abc.run(minimum_epsilon=.1, max_nr_populations=12,)

# %%
plot_kde = False
if plot_kde:
    fig, ax = plt.subplots()
    for t in range(history.max_t+1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df, w,
            xmin=0, xmax=1,
            x="mu_go", ax=ax,
            label="PDF t={}".format(t))
    #ax.axvline(observed_presp.presp.values, color="k", linestyle="dashed");
    ax.legend();


# %%
plot_ci = False
if plot_ci:
    ci_ax = pyabc.visualization.plot_credible_intervals(history)
    def get_map_estimates(ci_ax):
        map_estimates = {}
        for ax in ci_ax:
            map_estimates[ax.get_ylabel()] = ax.get_lines()[0].get_ydata()[-1]
        return(map_estimates)
    map_estimates = get_map_estimates(ci_ax)
    map_estimates

# %%

simulation = stopsignal_model(map_estimates)
print(simulation)

plot_presp = False
if plot_presp:
    plt.plot(simulation['presp'])
    plt.plot(observed_presp.presp.values, 'k')


# %%

# %%
params

# %%

# %%
