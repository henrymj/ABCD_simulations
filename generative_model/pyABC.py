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

# %%
# create the main model function
# takes in a dict of model parameters
# returns a dict of peformance statistics

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
    # need to separate presp values since distance fn can't take a vector
    for i, value in enumerate(stop_data):
        results[f'presp_{i}'] = value

    return(results)

parameter_prior = Distribution(mu_go=RV("uniform", 0, .5),
                               mu_stop_delta=RV("uniform", 0, 1),
                              mu_delta_incorrect=RV("uniform", 0, 0.5),
                              noise_sd=RV("uniform", 2, 5),
                              nondecision=RV("uniform", 25, 75))
parameter_prior.get_parameter_names()


# %%
# testing the model function to make sure it works...

test_model = False
if test_model:
    params={'mu_delta_incorrect': 0.10386248711279868,
    'mu_go': 0.11422675271126799,
    'mu_stop_delta': 0.7850423871897488,
    'noise_sd': 3.1238287051634597,
    'nondecision':50}

    simulation = stopsignal_model(params)
    simulation


# %%
# use an adaptive distance function
# which deals with the fact that different outcome measures have
# different scales - automatically scales to them

distance_adaptive = pyabc.AdaptivePNormDistance(p=2)

abc = ABCSMC(stopsignal_model, parameter_prior, distance_adaptive)


# %%
# set up the database for the simulation
db_path = pyabc.create_sqlite_db_id(file_="./adaptive_distance.db")

# observed metrics specified here by hand.  should instead store metrics to json
# and load those
observed_data = {'mean_go_RT': 455.367, 'mean_stopfail_RT': 219.364, 'go_acc': .935}

# load presp data from txt file
observed_presp = pd.read_csv('presp_by_ssd_inperson.txt',  delimiter=r"\s+", index_col=0)
for i, value in enumerate(observed_presp.presp.values):
    observed_data[f'presp_{i}'] = value

# initiatize database and add observed data
abc.new(db_path, observed_data)

# %%
# run the model
history = abc.run(minimum_epsilon=.1, max_nr_populations=15,)
