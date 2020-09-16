import numpy as np
import pandas as pd
import argparse
from os import path
from sympy.solvers import solve
from sympy import Symbol

from simulate_individual import generate_exgauss_sampler_from_fit

from utils import SimulateData


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulations')
    parser.add_argument('--n_trials', default=1000)
    parser.add_argument('--method', default='vanilla',
                        help='choose from vanilla, guesses, log, linear')
    parser.add_argument('--abcd_dir', default='./abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--out_dir', default='./simulated_data',
                        help='location to save simulated data')
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':
    args = get_args()

    # GET ABCD INFO
    abcd_data = pd.read_csv('%s/minimal_abcd_no_issue_3.csv' % args.abcd_dir)

    SSDs = abcd_data.SSDDur.unique()
    SSDs = [i for i in SSDs if i == i]
    SSDs.sort()
    acc_per_SSD = pd.DataFrame()
    for ssd in SSDs:
        curr_means = abcd_data.query(
            "SSDDur == %s and correct_stop==0.0" % ssd
        ).groupby('NARGUID').mean()['choice_accuracy']
        curr_means.name = ssd
        acc_per_SSD = pd.concat([acc_per_SSD, curr_means], 1, sort=True)

    go_accs = abcd_data.query(
            "trial_type == 'GoTrial' and correct_go_response in ['1.0', '0.0']"
        ).groupby('NARGUID').mean()['choice_accuracy']
    go_accs.name = -1
    acc_per_SSD = pd.concat([acc_per_SSD, go_accs], 1, sort=True)

    p = Symbol('p')
    guess_mean = acc_per_SSD.mean()[0.0]
    go_mean = acc_per_SSD.mean()[-1]
    p_guess_per_SSD = []
    for ssd in acc_per_SSD.columns:
        curr_mean = acc_per_SSD.mean()[ssd]
        solution = solve(p*guess_mean + (1-p)*go_mean - curr_mean, p)
        assert len(solution) == 1
        p_guess_per_SSD.append(solution[0])

    p_guess_dict = {SSD: p for SSD, p in zip(SSDs, p_guess_per_SSD)}

    SSD0_RTs = abcd_data.query(
        "SSDDur == 0.0 and correct_stop==0.0"
        ).stop_rt_adjusted.values
    sample_exgauss = generate_exgauss_sampler_from_fit(SSD0_RTs)

    SSD_dist_per_sub = pd.read_csv('%s/SSD_dist_by_subj.csv' % args.abcd_dir)

    # SIMULATE
    simulator_dict = {
        'vanilla': SimulateData(),
        'guesses': SimulateData(guesses=True),
        'log': SimulateData(mu_go_grader='log'),
        'linear': SimulateData(mu_go_grader='linear')
    }
    simulator = simulator_dict[args.method]

    out_file = '%s/ABCD_SSD_group_%s.csv' % (args.out_dir, args.method)
    if path.exists(out_file):
        group_data = pd.read_csv(out_file, index_col=0)
        remaining_subs = list(set(SSD_dist_per_sub.NARGUID.unique()) -
                              set(group_data.ID.unique()))
    else:
        group_data = pd.DataFrame()
        remaining_subs = SSD_dist_per_sub.NARGUID.unique()

    for subject in remaining_subs:
        sub_df = SSD_dist_per_sub.query("NARGUID=='%s'" % subject)
        sub_SSDs = sub_df.SSDDur.values
        params = {
            'n_trials_go': args.n_trials,
            'n_trials_stop': [int(i) for i in np.rint(
                sub_df.proportion.values * args.n_trials)],
            'SSDs': sub_SSDs,
            'guess_function': sample_exgauss,
            'p_guess': [p_guess_dict[ssd] for ssd in sub_SSDs],
        }
        data = simulator.simulate(params)
        data['ID'] = subject
        group_data = pd.concat([group_data, data], 0)
        group_data.to_csv(out_file)
