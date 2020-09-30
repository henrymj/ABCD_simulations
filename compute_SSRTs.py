import numpy as np
import pandas as pd
import argparse
from os import path
from glob import glob
import scipy.stats as sstats

from stopsignalmetrics import SSRTmodel
from utils import SimulateData
from simulate import generate_exgauss_sampler_from_fit


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulations')
    parser.add_argument('--abcd_dir', default='./abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--sim_dir', default='./simulated_data',
                        help='location of simulated data')
    parser.add_argument('--out_dir', default='./ssrt_metrics',
                        help='location to save ssrt metrics')
    parser.add_argument('--n_graded_go_trials', default=20000)
    args = parser.parse_args()
    return(args)


def generate_out_df(data, SSD_guess_dict, graded_go_dict):
    info = []
    ssrtmodel = SSRTmodel(model='replacement')
    goRTs = data.loc[data.goRT.notnull(), 'goRT'].values
    SSDs = [i for i in data.SSD.unique() if i == i]
    SSDs.sort()

    for SSD in SSDs:
        curr_df = data.query(
            "condition=='go' | condition=='stop' and SSD == %s" % SSD
            ).copy()
        curr_metrics = ssrtmodel.fit_transform(curr_df)
        if (curr_metrics['p_respond'] == 0) | (curr_metrics['p_respond'] == 1):
            curr_info = [v for v in curr_metrics.values()] +\
                    [SSD, np.nan, np.nan]
        else:
            goRTs_w_guesses = add_guess_RTs_and_sort(goRTs,
                                                     SSD,
                                                     SSD_guess_dict)
            if SSD < 200:
                print('w guesses:')
            SSRT_w_guesses = SSRT_wReplacement(curr_metrics,
                                               goRTs_w_guesses,
                                               verbose=(SSD < 200))
            if SSD < 200:
                print('w graded:')
            SSRT_w_graded = SSRT_wReplacement(curr_metrics,
                                              graded_go_dict[SSD].copy(),
                                              verbose=(SSD < 200))

            curr_info = [v for v in curr_metrics.values()] +\
                        [SSD, SSRT_w_guesses, SSRT_w_graded]
        info.append(curr_info)
        cols = [k for k in curr_metrics.keys()] +\
               ['SSD', 'SSRT_w_guesses', 'SSRT_w_graded']

    return pd.DataFrame(
        info,
        columns=cols)


def add_guess_RTs_and_sort(goRTs, SSD, SSD_guess_dict):
    curr_n = len(goRTs)
    p_guess = SSD_guess_dict[SSD]
    if p_guess == 1.0:
        guess_RTs = sample_exgauss(curr_n)
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
        guess_RTs = sample_exgauss(n_guess)
        all_RTs = np.concatenate([goRTs, guess_RTs])
        all_RTs.sort()
        return all_RTs


def simulate_graded_RTs_and_sort(n_trials, SSD, verbose=False):
    simulator = SimulateData()
    params = simulator._init_params({})
    params['n_trials_stop'] = n_trials
    params['n_trials_go'] = n_trials

    params['mu_go'] = simulator._log_mu_go(params['mu_go'], SSD)
    simulator._set_n_trials(params)
    simulator._set_n_guesses(params)  # no guessing is happening

    data_dict = simulator._simulate_go_trials(simulator._init_data_dict(),
                                              params)
    goRTs = data_dict['RT'].copy()
    goRTs.sort()
    if verbose:
        print(SSD)
        for p in np.arange(0, 100, 5):
            print(p, sstats.scoreatpercentile(goRTs, p))
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


def SSRT_wReplacement(metrics, sorted_go_RTs, verbose=False):
    P_respond = metrics['p_respond']
    goRTs_w_replacements = np.concatenate((
        sorted_go_RTs,
        [metrics['max_RT']] * metrics['omission_count']))
    goRTs_w_replacements.sort()
    nrt = get_nth_RT(P_respond, goRTs_w_replacements)
    if verbose:
        print('SSD', metrics['mean_SSD'])
        print('p_respond', P_respond)
        print('nrt', nrt)
    return nrt - metrics['mean_SSD']


if __name__ == '__main__':
    args = get_args()

    # GET ABCD INFO
    abcd_data = pd.read_csv('%s/minimal_abcd_clean.csv' % args.abcd_dir)
    p_guess_df = pd.read_csv('%s/p_guess_per_ssd.csv' % args.abcd_dir)

    SSD_guess_dict = {float(col): float(p_guess_df[col].values[0]) for col
                      in p_guess_df.columns}
    print(SSD_guess_dict)

    SSD0_RTs = abcd_data.query(
        "SSDDur == 0.0 and correct_stop==0.0"
        ).stop_rt_adjusted.values
    sample_exgauss = generate_exgauss_sampler_from_fit(SSD0_RTs)

    # SET UP GRADED MU GO DISTS
    graded_go_dict = {}
    for SSD in [i for i in abcd_data.SSDDur.unique() if i == i]:
        graded_go_dict[SSD] = simulate_graded_RTs_and_sort(
            args.n_graded_go_trials,
            SSD,
            verbose=(SSD < 200))

    # CALCULATE SSRT
    for data_file in glob(path.join(args.sim_dir, '*.csv')):
        sim_type = path.basename(
            data_file
            ).replace('.csv', '')
        out_df = generate_out_df(pd.read_csv(data_file),
                                 SSD_guess_dict,
                                 graded_go_dict)
        out_df.to_csv(path.join(args.out_dir, '%s.csv' % sim_type))
