import pandas as pd
import numpy as np
import argparse
import json
from os import path
from glob import glob
from simulate_individuals import generate_exgauss_sampler_from_fit,\
                                 generate_exgauss_sampler_from_params,\
                                 get_SSDs
from stopsignalmetrics import SSRTmodel
from utils import SimulateData


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulations')
    parser.add_argument('--n_graded_go_trials', default=5000)
    parser.add_argument('--mu_suffix', required=True)
    parser.add_argument('--subjects', nargs='+',
                        help='subjects to run simulations on', required=True)
    parser.add_argument('--abcd_dir', default='../abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--sim_dir_base',
                        default='../simulated_data/individual_data',
                        help='location to save simulated data')
    parser.add_argument('--out_dir_base',
                        default='../ssrt_metrics/individual_metrics',
                        help='location to save simulated data')
    parser.add_argument('--clip_SSDs_bool',
                        default=True,
                        help='clip fixed SSD design to clipped_SSD instead of max',
                        type=bool)
    parser.add_argument('--clipped_SSD',
                        default=500,
                        help='max SSD to use if dist is clipped')
    parser.add_argument('--max_SSD',
                        default=900,
                        help='max SSD of the dataset')
    return(parser.parse_args())


def generate_out_df(data,
                    SSD_guess_dict,
                    graded_go_dict,
                    guess_sampler,
                    tracked_data):
    info = []
    ssrtmodel = SSRTmodel(model='replacement')
    goRTs = data.loc[data.goRT.notnull(), 'goRT'].values
    SSDs = [i for i in data.SSD.unique() if i == i]
    SSDs.sort()

    for SSD in SSDs:
        curr_df = data.query(
            "condition=='go' | (condition=='stop' and SSD == %s)" % SSD
            )
        curr_metrics = ssrtmodel.fit_transform(curr_df)
        if (curr_metrics['p_respond'] == 0) | (curr_metrics['p_respond'] == 1):
            curr_info = [v for v in curr_metrics.values()] +\
                    [SSD, np.nan, np.nan]
        else:
            goRTs_w_guesses = add_guess_RTs_and_sort(goRTs,
                                                     SSD,
                                                     SSD_guess_dict,
                                                     guess_sampler)
            SSRT_w_guesses = SSRT_wReplacement(curr_metrics,
                                               goRTs_w_guesses)
            SSRT_w_graded = SSRT_wReplacement(curr_metrics,
                                              graded_go_dict[SSD].copy())

            curr_info = [v for v in curr_metrics.values()] +\
                        [SSD, SSRT_w_guesses, SSRT_w_graded]
        info.append(curr_info)
    cols = [k for k in curr_metrics.keys()] +\
           ['SSD', 'SSRT_w_guesses', 'SSRT_w_graded']
    # get for metrics using "whole" simulated data
    curr_data = data.loc[data.SSD.isin(np.arange(50, 300, 50)) | (data.condition=='go')]
    curr_metrics = ssrtmodel.fit_transform(curr_data)
    curr_info = [v for v in curr_metrics.values()] +\
                [-np.inf, np.nan, np.nan]
    info.append(curr_info)

    # get metrics for tracking-based data
    # double kludge, -inf for fixed, +inf for tracking
    tracked_metrics = ssrtmodel.fit_transform(tracked_data)
    tracked_info = [v for v in tracked_metrics.values()] +\
                   [np.inf, np.nan, np.nan]
    info.append(tracked_info)

    return pd.DataFrame(
        info,
        columns=cols)


def add_guess_RTs_and_sort(goRTs, SSD, SSD_guess_dict, guess_sampler):
    curr_n = len(goRTs)
    p_guess = SSD_guess_dict[SSD]
    if p_guess == 1.0:
        guess_RTs = guess_sampler(curr_n)
        guess_RTs.sort()
        return guess_RTs
    elif p_guess <= 0:  # SSDs 550 and 650
        goRTs.sort()
        return goRTs
    else:
        # Equation logic:
        # p_guess = n_guess / (n_guess + curr_n) =>
        # n_guess = (p_guess * curr_n) / (1 - p_guess)
        n_guess = int(np.rint(float((p_guess*curr_n)/(1-p_guess))))
        guess_RTs = guess_sampler(n_guess)
        all_RTs = np.concatenate([goRTs, guess_RTs])
        all_RTs.sort()
        return all_RTs


def simulate_graded_RTs_and_sort(n_trials, SSD, sub_params=None):
    sub_params = sub_params.copy() if sub_params else {}
    simulator = SimulateData()
    params = simulator._init_params(sub_params)
    params['n_trials_stop'] = 0
    params['n_trials_go'] = n_trials

    params['mu_go'] = simulator._log_grade_mu(params['mu_go_base'], SSD)
    simulator._set_n_trials(params)
    simulator._set_n_guesses(params)  # no guessing is happening

    data_dict = simulator._simulate_go_trials(simulator._init_data_dict(),
                                              params)
    # copy, drop nans, sort
    goRTs = [rt for rt in data_dict['RT'] if rt == rt]
    goRTs.sort()
    return goRTs


def get_nth_RT(P_respond, goRTs):
    """Get nth RT based P(response|signal) and sorted go RTs."""
    nth_index = int(np.rint(P_respond*len(goRTs))) - 1
    if nth_index < 0:
        nth_RT = goRTs[0]
    elif nth_index >= len(goRTs):
        nth_RT = goRTs[-1]
    else:
        nth_RT = goRTs[nth_index]
    return nth_RT


def SSRT_wReplacement(metrics, sorted_go_RTs):
    P_respond = metrics['p_respond']
    goRTs_w_replacements = np.concatenate((
        sorted_go_RTs,
        [metrics['max_RT']] * metrics['omission_count']))
    goRTs_w_replacements.sort()
    nrt = get_nth_RT(P_respond, goRTs_w_replacements)
    return nrt - metrics['mean_SSD']


if __name__ == '__main__':
    args = get_args()

    # GET ABCD INFO
    p_guess_df = pd.read_csv('%s/p_guess_per_ssd.csv' % args.abcd_dir)

    SSD_guess_dict = {float(col): float(p_guess_df[col].values[0]) for col
                      in p_guess_df.columns}

    # assigned mus
    with open('%s/assigned_mus_%s.json' % (args.abcd_dir, args.mu_suffix)) as json_file:
        mus_dict = json.load(json_file)

    # exgaus sampler for guesses
    exgauss_param_path = '%s/exgauss_params.json' % args.abcd_dir
    with open(exgauss_param_path, 'r') as f:
        exgauss_params = json.load(f)
    sample_exgauss = generate_exgauss_sampler_from_params(exgauss_params)

    # CALCULATE SSRT
    SSDs = get_SSDs(args)
    issue_subs = []

    sim_dir = args.sim_dir_base + '_' + args.mu_suffix
    out_dir = args.out_dir_base + '_' + args.mu_suffix

    for sub in args.subjects:
        try:
            params = {
                'mu_go_base': mus_dict[sub]['go'],
                'mu_stop': mus_dict[sub]['stop']
            }
            graded_go_dict = {}
            for SSD in SSDs:
                graded_go_dict[SSD] = simulate_graded_RTs_and_sort(
                    args.n_graded_go_trials,
                    SSD,
                    sub_params=params)

            for data_file in glob(path.join(sim_dir, 'fixed', '*%s*.csv' % sub)):
                sim_type = path.basename(
                    data_file
                    ).replace('.csv', '')
                tracked_data_file = data_file.replace('fixed', 'tracking')
                out_df = generate_out_df(pd.read_csv(data_file),
                                         SSD_guess_dict,
                                         graded_go_dict,
                                         sample_exgauss,
                                         pd.read_csv(tracked_data_file))
                out_df.to_csv(path.join(out_dir, '%s.csv' % sim_type))
        except KeyError as err:
            print("KeyError error for sub {0}: {1}".format(sub, err))
            issue_subs.append(sub)
            continue
    if len(issue_subs) > 0:
        print('issue subs: ', issue_subs)
    else:
        print('no problematic subs run here!')
