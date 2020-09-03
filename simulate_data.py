import numpy as np
import pandas as pd
import argparse
from sympy.solvers import solve
from sympy import Symbol
import scipy.stats as sstats

from utils import SimulateData


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulations')
    parser.add_argument('--n_subjects', default=250)
    parser.add_argument('--n_trials', default=650)
    parser.add_argument('--out_dir', default='./Simulated_Data',
                        help='location to save simulated data')
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
        return (exp_out+norm_out) + loc

    return sample_exgauss


if __name__ == '__main__':
    args = get_args()

    # GET ABCD INFO
    abcd_data = pd.read_csv('minimal_abcd_no_issue_3.csv')

    SSDs = abcd_data.SSDDur.unique()
    SSDs = [i for i in SSDs if i == i and i <= 550]
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

    SSD0_RTs = abcd_data.query(
        "SSDDur == 0.0 and correct_stop==0.0"
        ).stop_rt_adjusted.values
    sample_exgauss = generate_exgauss_sampler_from_fit(SSD0_RTs)

    # SIMULATE
    subjects = np.arange(0, args.n_subjects)

    simulator_dict = {
        'vanilla': SimulateData(),
        'guesses': SimulateData(guesses=True),
        'log': SimulateData(graded_mu_go=True),
        'linear': SimulateData(graded_mu_go=True)
    }

    group_data_dict = {
        'vanilla': pd.DataFrame(),
        'guesses': pd.DataFrame(),
        'log': pd.DataFrame(),
        'linear': pd.DataFrame(),
    }

    for subject in subjects:
        params = {
            'n_trials': args.n_trials,
            'SSDs': SSDs,
            'mu_go': np.random.normal(.25, scale=.05),
            'mu_stop': np.random.normal(.6, scale=.05),
            'guess_function': sample_exgauss,
            'p_guess': p_guess_per_SSD,
        }
        for sim_key in simulator_dict:
            params['mu_go_grader'] = sim_key
            data = simulator_dict[sim_key].simulate(params)
            data['ID'] = subject
            group_data_dict[sim_key] = pd.concat(
                [group_data_dict[sim_key], data],
                0)

    for sim_key in group_data_dict:
        curr_group = group_data_dict[sim_key].copy()
        curr_group['simulation'] = sim_key
        curr_group.to_csv('%s/%s.csv' % (args.out_dir, sim_key))
