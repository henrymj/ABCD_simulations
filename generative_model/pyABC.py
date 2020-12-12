# ## Likelihood-free estimation of stop model parameters
#
# This notebook shows how to use approximate Bayesian computation via sequential Monte Carlo (ABC-SMC) using the pyABC package.

from stopsignalmodel import StopSignalModel
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


def get_args():
    parser = argparse.ArgumentParser(description='fit stop model params using ABC-SMC')
    parser.add_argument('--study', help='study label to define dataset to be fit')
    parser.add_argument('--model', nargs='+', help='model label(s)', required=True)
    parser.add_argument('--generator', help='model used to generate data (for model recovery')
    parser.add_argument('--generative_paramfile', help='params file for generative model', default='params/generative_model.json')
    parser.add_argument('--nthreads', help='max # of multiprocessing threads', default=8)
    parser.add_argument('--max_populations', help='max # of ABC populations', default=12)
    parser.add_argument('--min_epsilon', help='epsilon threshold for ABC', default=.1)
    parser.add_argument('--min_ssd', help='minimum SSD value', default=0)
    parser.add_argument('--max_ssd', help='maximum SSD value', default=500)
    parser.add_argument('--ssd_step', help='SSD step size', default=50)
    parser.add_argument('--random_seed', help='random seed', type=int)
    parser.add_argument('--p_guess_go', help='p_guess on go trials (default None)')
    parser.add_argument('--tempdb', help='use a temporary db file', action='store_true')
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


def get_parameter_priors(parameters):
    use_guessing = 'pguess' in parameters and parameters['pguess'] is not None
    if not use_guessing:
        return Distribution(
            mu_go=RV("uniform", 0.1, 1),
            mu_stop_delta=RV("uniform", 0, 1),
            mu_delta_incorrect=RV("uniform", 0, 0.8),
            noise_sd=RV("uniform", 0, 5),
            nondecision=RV("uniform", 0, 100))
    else:
        return Distribution(
            mu_go=RV("uniform", 0.1, 1),
            mu_stop_delta=RV("uniform", 0, 1),
            mu_delta_incorrect=RV("uniform", 0, 0.8),
            noise_sd=RV("uniform", 0, 5),
            nondecision=RV("uniform", 0, 100),
            pguess=RV("uniform", 0., .3))


def get_parameters(models):
    params = {}
    for model in models:
        paramfile = f'params/params_{model}.json'
        with open(paramfile) as f:
            params[model] = json.load(f)
    return(params)


if __name__ == '__main__':
    args = get_args()
    if args.study is not None:
        print(f'fitting stop task for {args.study}')
    elif args.generator is not None:
        print(f'fitting stop task for data generated by {args.generator}')
    else:
        raise Exception('You must specify either study or generator')

    # make sure there are no duplicated models
    assert len(args.model) == len(set(args.model)), "all models must be unique"

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

    parameters = get_parameters(args.model)

    use_guessing = {}
    # first need to set up the model instances, so that their
    # fit_transform() function can be passed below
    stopsignalmodels = {
        p: StopSignalModel(p, parameters=parameters[p]) for p in parameters
    }

    # set up priors amd models for ABC
    # these values are based on some hand-tweaking using the in-person dataset
    stopsignal_model_func = [stopsignalmodels[x].fit_transform for x in args.model]
    parameter_prior = [get_parameter_priors(parameters) for x in args.model]
    if len(args.model) == 1:  # expects a single item, not a list

        stopsignal_model_func = stopsignal_model_func[0]
        parameter_prior = parameter_prior[0]

    if args.verbose:
        print(parameter_prior)

    # set up generative model
    if args.generator is None:
        descriptor = args.study
        observed_data = get_observed_data(args)
    else:
        descriptor = f'generative_{args.generator}'
        if args.generative_paramfile is None:
            generative_paramfile = 'params/generative_model.json'
        else:
            generative_paramfile = args.generative_paramfile
        with open(generative_paramfile) as f:
            generative_model_params = json.load(f)

        if args.generator in ['basic', 'gradedmugo']:
            del generative_model_params['pguess']
        generative_model = StopSignalModel(
            args.generator,
            paramfile=f'params/params_{args.generator}.json')
        observed_data = generative_model.fit_transform(generative_model_params)

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

    abc = ABCSMC(stopsignal_model_func, parameter_prior, distance,
                 acceptor=pyabc.UniformAcceptor(use_complete_history=True),
                 stop_if_only_single_model_alive=True)

    # set up the database for the simulation
    if args.tempdb:
        print('Test mode: using temp database')
        db_path = (
            f"sqlite:///{os.path.join(tempfile.gettempdir(), 'test.db')}")
    else:
        modelname = '_'.join(args.model)
        distance_string = 'fixed_distance' if args.fixed_distance else 'adaptive_distance'
        db_path = f'sqlite:///results/{descriptor}_{modelname}_{distance_string}.db'

    # initiatize database and add observed data
    abc.new(db_path, observed_data)

    if not args.setuponly:
        sampler_history = abc.run(
            minimum_epsilon=min_epsilon,
            max_nr_populations=max_populations,)
