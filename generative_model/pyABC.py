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
    parser.add_argument('--model', nargs='+', help='model label(s)', required=True)
    parser.add_argument('--generator',  help='model used to generate data (for model recovery')
    parser.add_argument('--generator_params',  help='params file for generative model', default='params/generative_model.json')
    parser.add_argument('--nthreads', help='max # of multiprocessing threads', default=8)
    parser.add_argument('--max_populations', help='max # of ABC populations', default=12)
    parser.add_argument('--min_epsilon', help='epsilon threshold for ABC', default=.1)
    parser.add_argument('--min_ssd', help='minimum SSD value', default=0)
    parser.add_argument('--max_ssd', help='maximum SSD value', default=500)
    parser.add_argument('--ssd_step', help='SSD step size', default=50)
    parser.add_argument('--random_seed', help='random seed', type=int)
    parser.add_argument('--p_guess_go', help='p_guess on go trials (default None)')
    parser.add_argument('--test', help='use a test db file', action='store_true')
    parser.add_argument('--setuponly', help='do not run estimation', action='store_true')
    parser.add_argument('--fixed_distance', help='use fixed rather than adaptive distance',
                        action='store_true')
    parser.add_argument('--debug', help='turn on debugging output from ABC', action='store_true')
    parser.add_argument('--verbose', help='turn on verbose output', action='store_true')
    parser.add_argument('--stop_guess_ABCD', help='use SSD-dependent guessing based on ABCD data', action='store_true')

    parser.add_argument('--guess_param_file', default='exgauss_params.json',
                        help='file with exgauss params for guesses')
    parser.add_argument('--tracking', help='use tracking algorithm', action='store_true')
    return parser.parse_args()


# create a single-layer metrics dict for output
# since the pickler can't handle multilevel dicts
def cleanup_metrics(metrics):
    for k in metrics['SSRT']:
        metrics['SSRT_' + k] = metrics['SSRT'][k]
    del metrics['SSRT']
    return(metrics)


def setup_ssd_params(params):
    # set default params if not present
    if 'min_ssd' not in params:
        params['min_ssd'] = 0
    if 'max_ssd' not in params:
        params['max_ssd'] = 500
    if 'ssd_step' not in params:
        params['ssd_step'] = 50
    return(params)


def stopsignal_model_gradedmugo(parameters):
    """wrapper for graded mu go model

    Args:
        parameters (dict): parameters for model
    """
    # load full initial parameter set from file
    paramfile = f'params/params_gradedmugo.json'
    with open(paramfile) as f:
        params = json.load(f)

    # install ABCSMC parameters into full parameter set
    parameters['nondecision'] = int(parameters['nondecision'])
    params['mu']['go'] = parameters['mu_go']
    params['mu_go_grader'] = "log"
    params['mu']['stop'] = parameters['mu_go'] + parameters['mu_stop_delta']
    params['mu_delta_incorrect'] = parameters['mu_delta_incorrect']
    params['noise_sd'] = {'go': parameters['noise_sd'],
                          'stop': parameters['noise_sd']}
    params['nondecision'] = {'go': parameters['nondecision'],
                             'stop': parameters['nondecision']}
    params = setup_ssd_params(params)
    return(_stopsignal_model(params))


def stopsignal_model_gradedmuboth(parameters):
    """wrapper for graded mu go model

    Args:
        parameters (dict): parameters for model
    """
    # load full initial parameter set from file
    paramfile = f'params/params_gradedmugo.json'
    with open(paramfile) as f:
        params = json.load(f)

    # install ABCSMC parameters into full parameter set
    parameters['nondecision'] = int(parameters['nondecision'])
    params['mu']['go'] = parameters['mu_go']
    params['mu_go_grader'] = "log"
    params['mu_stop_grader'] = "log"
    params['mu']['stop'] = parameters['mu_go'] + parameters['mu_stop_delta']
    params['mu_delta_incorrect'] = parameters['mu_delta_incorrect']
    params['noise_sd'] = {'go': parameters['noise_sd'],
                          'stop': parameters['noise_sd']}
    params['nondecision'] = {'go': parameters['nondecision'],
                             'stop': parameters['nondecision']}
    params = setup_ssd_params(params)
    return(_stopsignal_model(params))


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
    params = setup_ssd_params(params)
    return(_stopsignal_model(params))


def stopsignal_model_scaledguessing(parameters):
    """wrapper for scaled guessing model using ABCD estimates

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
                         'stop': 'ABCD'}
    params = setup_ssd_params(params)
    return(_stopsignal_model(params))


def stopsignal_model_fullabcd(parameters):
    """wrapper for scaled guessing model + grade mu go using ABCD estimates

    Args:
        parameters (dict): parameters for model
    """
    # load full initial parameter set from file
    paramfile = f'params/params_fullabcd.json'
    with open(paramfile) as f:
        params = json.load(f)

    # install ABCSMC parameters into full parameter set
    parameters['nondecision'] = int(parameters['nondecision'])
    params['mu_go_grader'] = "log"
    params['mu']['go'] = parameters['mu_go']
    params['mu']['stop'] = parameters['mu_go'] + parameters['mu_stop_delta']
    params['mu_delta_incorrect'] = parameters['mu_delta_incorrect']
    params['noise_sd'] = {'go': parameters['noise_sd'],
                          'stop': parameters['noise_sd']}
    params['nondecision'] = {'go': parameters['nondecision'],
                             'stop': parameters['nondecision']}
    params['p_guess'] = {'go': parameters['pguess'],
                         'stop': 'ABCD'}
    params = setup_ssd_params(params)
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
    params = setup_ssd_params(params)
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

    ssd = fixedSSD(
        np.arange(params['min_ssd'],
                  params['max_ssd'] + 1,  # add 1 to include max
                  params['ssd_step']))

    study = StopTaskStudy(ssd, None, params=params)

    trialdata = study.run()
    trialdata['correct'] = trialdata.correct.astype(float)
    metrics = study.get_stopsignal_metrics()
    # summarize data - go trials are labeled with SSD of -inf so that
    # they get included in the summary
    presp_by_ssd = trialdata.groupby('SSD').mean().query('SSD >= 0').resp.values
    results = {}

    metrics = cleanup_metrics(metrics)
    for k in ['mean_go_RT', 'mean_stopfail_RT', 'go_acc', 'sd_go_RT', 'sd_stopfail_RT']:
        results.update({k: metrics[k]})
    # need to separate presp values since distance fn can't take a vector
    for i, value in enumerate(presp_by_ssd):
        # occasionally there will be no trials for a particular SSD which gives NaN
        # we replace that with zero
        results[f'presp_{i}'] = 0 if np.isnan(value) else value

    for i, SSD in enumerate(trialdata.query('SSD >= 0').SSD.sort_values().unique()):
        accdata_for_ssd = trialdata.query(f'SSD == {SSD}').dropna()
        value = accdata_for_ssd.correct.dropna().mean()
        results[f'accuracy_{i}'] = 0 if np.isnan(value) else value
    return(results)


def get_observed_data(args):
    # load the data to be fitted
    # this should contain the output from stopsignalmetrics, including:
    # {'mean_go_RT': , 'mean_stopfail_RT': , 'go_acc': }
    with open(f'data/data_{args.study}.json') as f:
        observed_data = json.load(f)

    # load presp data from txt file
    observed_presp = pd.read_csv(f'data/presp_by_ssd_{args.study}.txt',
                                 delimiter=r"\s+", index_col=0)
    assert 'presp' in observed_presp.columns and observed_presp.index.name == 'SSD', 'presp file must include column presp and SSD as index'
    observed_presp = observed_presp[observed_presp.index <= args.max_ssd]
    observed_presp = observed_presp[observed_presp.index >= args.min_ssd]
    for i, value in enumerate(observed_presp.presp.values):
        observed_data[f'presp_{i}'] = value

    # load presp data from txt file
    observed_accuracy = pd.read_csv(f'data/accuracy_by_ssd_{args.study}.txt',
                                    delimiter=r"\s+", index_col=0)
    assert 'accuracy' in observed_accuracy.columns and observed_accuracy.index.name == 'SSD', 'accuracy file must include column accuuracy and SSD as index'
    observed_accuracy = observed_accuracy[observed_accuracy.index <= args.max_ssd]
    observed_accuracy = observed_accuracy[observed_accuracy.index >= args.min_ssd]
    for i, value in enumerate(observed_accuracy.accuracy.values):
        observed_data[f'accuracy_{i}'] = value if value is not None else 0

    return(observed_data)


def get_parameter_priors(model):
    priors = None
    if model in ['basic', 'gradedmugo', 'gradedmuboth']:
        priors = Distribution(
            mu_go=RV("uniform", .1, .5),
            mu_stop_delta=RV("uniform", 0, 1),
            mu_delta_incorrect=RV("uniform", 0, 0.2),
            noise_sd=RV("uniform", 2, 5),
            nondecision=RV("uniform", 25, 75))
    elif model in ['simpleguessing', 'scaledguessing', 'fullabcd']:
        priors = Distribution(
            mu_go=RV("uniform", .1, .5),
            mu_stop_delta=RV("uniform", 0, 1),
            mu_delta_incorrect=RV("uniform", 0, 0.2),
            noise_sd=RV("uniform", 2, 5),
            nondecision=RV("uniform", 25, 75),
            pguess=RV("uniform", 0., .5))
    else:
        raise Exception(f'priors not defined for model {model}')
    return(priors)


if __name__ == '__main__':
    args = get_args()
    if args.study is not None:
        print(f'fitting stop task for {args.study}')
    elif args.generator is not None:
        print(f'fitting stop task for data generated by {args.generator}')
    else:
        raise Exception('You must specify either study or generator')

    stopsignal_model_func = {
        'basic': stopsignal_model_basic,
        'simpleguessing': stopsignal_model_simpleguessing,
        'scaledguessing': stopsignal_model_scaledguessing,
        'gradedmugo': stopsignal_model_gradedmugo,
        'gradedmuboth': stopsignal_model_gradedmuboth,
        'fullabcd': stopsignal_model_fullabcd
    }
    for model in args.model:
        assert model in stopsignal_model_func

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

    # set up priors amd models for ABC
    # these values are based on some hand-tweaking using the in-person dataset
    if len(args.model) > 1:
        stopsignal_model = [stopsignal_model_func[x] for x in args.model]
        parameter_prior = [get_parameter_priors(x) for x in args.model]
    else:
        stopsignal_model = stopsignal_model_func[args.model[0]]
        parameter_prior = get_parameter_priors(args.model[0])

    if args.verbose:
        print(stopsignal_model)
        print(parameter_prior)

    if args.generator is None:
        descriptor = args.study
        observed_data = get_observed_data(args)
    else:
        descriptor = f'generative_{args.generator}'
        with open(args.generator_params) as f:
            generative_params = json.load(f)
        observed_data = stopsignal_model_func[args.generator](generative_params)

    # use an adaptive distance function
    # which deals with the fact that different outcome measures have
    # different scales - automatically scales to them

    if args.fixed_distance:
        distance = pyabc.PNormDistance(p=2)
    else:
        initial_weights = {k: 1 / (observed_data[k] * len(observed_data)) for k in observed_data}

        distance = pyabc.AdaptivePNormDistance(
            p=2, initial_weights=initial_weights,
            scale_function=pyabc.distance.root_mean_square_deviation)

    # set up the sampler
    # use acceptor which seems to improve performance with adaptive distance

    abc = ABCSMC(stopsignal_model, parameter_prior, distance,
                 acceptor=pyabc.UniformAcceptor(use_complete_history=True))

    # set up the database for the simulation
    if args.test:
        print('Test mode: using temp database')
        db_path = (
            f"sqlite:///{os.path.join(tempfile.gettempdir(), 'test.db')}")
    else:
        modelname = '_'.join(args.model)
        distance_string = 'fixed_distance' if args.fixed_distance else 'adaptive_distance'
        db_path = f'sqlite:///results/{descriptor}_{modelname}_{distance_string}.db'

    # initiatize database and add observed data
    abc.new(db_path, observed_data)

    # %%
    # run the model
    if not args.setuponly:
        sampler_history = abc.run(
            minimum_epsilon=min_epsilon,
            max_nr_populations=max_populations,)
