import json
import pandas as pd
import numpy as np
import argparse
import scipy.stats as sstats
from scipy.stats import exponnorm
from utils import SimulateData


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulations')
    parser.add_argument('--n_trials_stop', default=2500)
    parser.add_argument('--n_trials_tracking_stop', default=25000)
    parser.add_argument('--n_trials_go', default=5000)
    parser.add_argument('--subjects', nargs='+',
                        help='subjects to run simulations on', required=True)
    parser.add_argument('--abcd_dir',
                        default='../abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--out_dir_base',
                        default='../simulated_data/individual_data',
                        help='location to save simulated data')
    parser.add_argument('--clip_SSDs_bool',
                        default=True,
                        help='clip fixed SSD design to clipped SSD instead of max',
                        type=bool)
    parser.add_argument('--max_SSD',
                        default=900,
                        help='max SSD of the dataset')
    parser.add_argument('--clipped_SSD',
                        default=500,
                        help='max SSD to use if dist is clipped')
    args = parser.parse_args()
    return(args)


def generate_exgauss_sampler_from_fit(data,
                                      default_sample_size=100000):
    FIT_K, FIT_LOC, FIT_SCALE = sstats.exponnorm.fit(data)
    FIT_LAMBDA = 1/(FIT_K*FIT_SCALE)
    FIT_BETA = 1/FIT_LAMBDA

    def sample_exgauss(sample_size=default_sample_size,
                       beta=FIT_BETA, scale=FIT_SCALE, loc=FIT_LOC):
        exp_out = np.random.exponential(scale=beta, size=sample_size)
        norm_out = np.random.normal(scale=scale, size=sample_size)
        out = (exp_out+norm_out) + loc
        n_negatives = np.sum(out < 0)
        while n_negatives > 0:
            out[out < 0] = sample_exgauss(n_negatives,
                                          beta=beta,
                                          scale=scale,
                                          loc=loc)
            n_negatives = np.sum(out < 0)
        return out

    return sample_exgauss


def generate_exgauss_sampler_from_params(param_dict,
                                         default_size=100000):
    def sample_exgauss(size=default_size,
                       param_dict=param_dict):
        out = exponnorm.rvs(param_dict['K'],
                            param_dict['loc'],
                            param_dict['scale'],
                            size=size)
        n_negatives = np.sum(out < 0)
        while n_negatives > 0:
            out[out < 0] = sample_exgauss(n_negatives,
                                          param_dict=param_dict)
            n_negatives = np.sum(out < 0)
        return out

    return sample_exgauss


def get_SSDs(args):
    max_SSD = args.clipped_SSD if args.clip_SSDs_bool else args.max_SSD
    return np.arange(0, max_SSD+50, 50)


if __name__ == '__main__':
    print('getting args')
    args = get_args()
    print('analyzing ABCD info')
    # GET ABCD INFO
    # p(guess | signal, SSD)
    p_guess_df = pd.read_csv('%s/p_guess_per_ssd.csv' % args.abcd_dir)
    p_guess_df.columns = p_guess_df.columns.astype(float)

    # exgaus sampler for guesses
    exgauss_param_path = '%s/exgauss_params.json' % args.abcd_dir
    with open(exgauss_param_path, 'r') as f:
        exgauss_params = json.load(f)
    sample_exgauss = generate_exgauss_sampler_from_params(exgauss_params)

    # assigned mus
    with open('%s/assigned_mus.json' % args.abcd_dir) as json_file:
        mus_dict = json.load(json_file)

    # SETUP SIMULATORS
    simulator_dict = {
        'standard': SimulateData(),
        'guesses': SimulateData(guesses=True),
        'graded_go': SimulateData(grade_mu_go=True),
        'graded_both': SimulateData(grade_mu_go=True, grade_mu_stop=True),
    }

    # set up shared params
    SSDs = get_SSDs(args)
    params = {
        'n_trials_stop': args.n_trials_stop,
        'n_trials_tracking_stop': args.n_trials_tracking_stop,
        'n_trials_go': args.n_trials_go,
        'guess_function': sample_exgauss,
        'SSDs': SSDs,
        'p_guess_stop': list(p_guess_df[SSDs].values.astype(float)[0])
    }

    # SIMULATE INDIVIDUALS
    issue_subs = []
    for sub in args.subjects:
        try:
            params['mu_go'] = mus_dict[sub]['go']
            params['mu_stop'] = mus_dict[sub]['stop']

            for sim_key in simulator_dict:
                for method in ['fixed', 'tracking']:
                    data = simulator_dict[sim_key].simulate(params,
                                                            method=method)
                    data['simulation'] = sim_key
                    data.to_csv('%s/%s/%s_%s.csv' % (args.out_dir_base,
                                                        method,
                                                        sim_key,
                                                        str(sub)))
        except KeyError as err:
            print("KeyError error for sub {0}: {1}".format(sub, err))
            issue_subs.append(sub)
            continue
    if len(issue_subs) > 0:
        print('issue subs: ', issue_subs)
    else:
        print('no problematic subs run here!')
