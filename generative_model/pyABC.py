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
import os
import pandas as pd
import logging
import argparse
import tempfile

from ssd import fixedSSD
from stoptaskstudy import StopTaskStudy


def get_args():
    parser = argparse.ArgumentParser(description='fit stop model params using ABC-SMC')
    parser.add_argument('--study', help='study label to define dataset to be fit')
    parser.add_argument('--model', help='model label')
    parser.add_argument('--nthreads', help='max # of multiprocessing threads', default=8)
    parser.add_argument('--max_populations', help='max # of ABC populations', default=12)
    parser.add_argument('--min_epsilon', help='epsilon threshold for ABC', default=.1)
    parser.add_argument('--min_ssd', help='minimum SSD value', default=0)
    parser.add_argument('--max_ssd', help='maximum SSD value', default=550)
    parser.add_argument('--ssd_step', help='SSD step size', default=50)
    parser.add_argument('--random_seed', help='random seed', type=int)
    parser.add_argument('--p_guess_go', help='p_guess on go trials (default None)')
    parser.add_argument('--test', help='use a test db file', action='store_true')
    parser.add_argument('--debug', help='turn on debugging output from ABC', action='store_true')
    parser.add_argument('--verbose', help='turn on verbose output', action='store_true')
    parser.add_argument('--stop_guess_ABCD', help='use SSD-dependent guessing based on ABCD data', action='store_true')

    parser.add_argument('--guess_param_file', default='exgauss_params.json',
                        help='file with exgauss params for guesses')
    # parser.add_argument('--tracking', help='use tracking algorithm', action='store_true')
    return parser.parse_args()


# create a single-layer metrics dict for output
# since the pickler can't handle multilevel dicts
def cleanup_metrics(metrics):
    for k in metrics['SSRT']:
        metrics['SSRT_' + k] = metrics['SSRT'][k]
    del metrics['SSRT']
    return(metrics)


def stopsignal_model_basic(parameters):
    """wrapper for basic model

    Args:
        parameters (dict): parameters for model
    """
    # load full initial parameter set from file
    paramfile = f'params/params_basic.json'
    with open(paramfile) as f:
        params = json.load(f)

    # install ABCSMC parameters into full parameter set
    parameters['nondecision'] = int(parameters['nondecision'])
    params['mu']['go'] = parameters['mu_go']
    params['mu']['stop'] = parameters['mu_go'] + parameters['mu_stop_delta']
    params['mu_delta_incorrect'] = parameters['mu_delta_incorrect']
    params['noise_sd'] = {'go': parameters['noise_sd'],
                          'stop': parameters['noise_sd']}
    params['nondecision'] = {'go': parameters['nondecision'],
                             'stop': parameters['nondecision']}
    return(_stopsignal_model(params))

def stopsignal_model_simpleguessing(parameters):
    """wrapper for simple guessing model

    Args:
        parameters (dict): parameters for model
    """
    # load full initial parameter set from file
    paramfile = f'params/params_simpleguessing.json'
    with open(paramfile) as f:
        params = json.load(f)

    # install ABCSMC parameters into full parameter set
    parameters['nondecision'] = int(parameters['nondecision'])
    params['mu']['go'] = parameters['mu_go']
    params['mu']['stop'] = parameters['mu_go'] + parameters['mu_stop_delta']
    params['mu_delta_incorrect'] = parameters['mu_delta_incorrect']
    params['noise_sd'] = {'go': parameters['noise_sd'],
                          'stop': parameters['noise_sd']}
    params['nondecision'] = {'go': parameters['nondecision'],
                             'stop': parameters['nondecision']}
    params['p_guess'] = {'go': parameters['pguess'],
                         'stop': parameters['pguess']}
    print(params)
    return(_stopsignal_model(params))


# create the main model function
# takes in a dict of model parameters
# returns a dict of peformance statistics
def _stopsignal_model(params):
    """[summary]

    Args:
        parameters (dict): model parameters from ABCSMC
        params (dict): starting parameters loaded from file
    """

    # install the parameters from the simulation
    # TBD
    #    if args.p_guess_file is not None:
    #        p_guess = pd.read_csv(args.p_guess_file, index_col=0)
    #        assert 'SSD' in p_guess.columns and 'p_guess' in p_guess.columns

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
    for k in ['mean_go_RT', 'mean_stopfail_RT', 'go_acc']:
        results.update({k: metrics[k]})
    # need to separate presp values since distance fn can't take a vector
    for i, value in enumerate(stop_data):
        results[f'presp_{i}'] = value

    return(results)


def get_observed_data(args):
    # load the data to be fitted
    # this should contain the output from stopsignalmetrics, including:
    # {'mean_go_RT': , 'mean_stopfail_RT': , 'go_acc': }
    with open(f'data/data_{args.study}.json') as f:
        observed_data = json.load(f)

    # observed_data = {'mean_go_RT': 455.367, 'mean_stopfail_RT': 219.364, 'go_acc': .935}

    # load presp data from txt file
    observed_presp = pd.read_csv(f'data/presp_by_ssd_{args.study}.txt',
                                 delimiter=r"\s+", index_col=0)
    for i, value in enumerate(observed_presp.presp.values):
        observed_data[f'presp_{i}'] = value
    return(observed_data)


def get_parameter_priors(model):
    priors = None
    if model == 'basic':
        priors = Distribution(
            mu_go=RV("uniform", .1, .5),
            mu_stop_delta=RV("uniform", 0, 1),
            mu_delta_incorrect=RV("uniform", 0, 0.2),
            noise_sd=RV("uniform", 2, 5),
            nondecision=RV("uniform", 25, 75))
    elif model == 'simpleguessing':
        priors = Distribution(
            mu_go=RV("uniform", .1, .5),
            mu_stop_delta=RV("uniform", 0, 1),
            mu_delta_incorrect=RV("uniform", 0, 0.2),
            noise_sd=RV("uniform", 2, 5),
            nondecision=RV("uniform", 25, 75),
            pguess=RV("uniform", 0., .5))
    return(priors)


if __name__ == '__main__':
    args = get_args()
    print(f'fitting stop task for {args.study}')

    if args.random_seed is not None:
        print('WARNING: Fixing random seed')
        np.random.seed(args.random_seed)

    # set numexpr thread limits
    os.environ['NUMEXPR_MAX_THREADS'] = str(args.nthreads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.nthreads)

    max_populations = int(args.max_populations)
    min_epsilon = float(args.min_epsilon)

    if args.debug:
        df_logger = logging.getLogger('Distance')
        df_logger.setLevel(logging.DEBUG)

    # set up priors for ABC
    # these values are based on some hand-tweaking using the in-person dataset
    parameter_prior = get_parameter_priors(args.model)

    if args.verbose:
        print(parameter_prior.get_parameter_names())

    # use an adaptive distance function
    # which deals with the fact that different outcome measures have
    # different scales - automatically scales to them

    distance_adaptive = pyabc.AdaptivePNormDistance(
        p=2, scale_function=pyabc.distance.root_mean_square_deviation)

    # set up the sampler
    # use acceptor which seems to improve performance with adaptive distance
    stopsignal_model = {
        'basic': stopsignal_model_basic,
        'simpleguessing': stopsignal_model_simpleguessing
    }

    abc = ABCSMC(stopsignal_model[args.model], parameter_prior, distance_adaptive,
                 acceptor=pyabc.UniformAcceptor(use_complete_history=True))

    # set up the database for the simulation
    if args.test:
        print('Test mode: using temp database')
        db_path = (
            f"sqlite:///{os.path.join(tempfile.gettempdir(), 'test.db')}")
    else:
        db_path = f'sqlite:///results/{args.study}_{args.model}_adaptive_distance.db'

    observed_data = get_observed_data(args)

    # initiatize database and add observed data
    abc.new(db_path, observed_data)

    # %%
    # run the model
    history = abc.run(
        minimum_epsilon=min_epsilon,
        max_nr_populations=max_populations,)
