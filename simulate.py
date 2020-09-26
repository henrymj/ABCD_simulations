import numpy as np
import pandas as pd
import argparse
import scipy.stats as sstats

from utils import SimulateData


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulations')
    parser.add_argument('--n_trials', default=125000)
    parser.add_argument('--abcd_dir', default='./abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--out_dir', default='./simulated_data',
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
        print('sampling')
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


if __name__ == '__main__':
    print('getting args')
    args = get_args()
    print('analyzing ABCD info')
    # GET ABCD INFO
    abcd_data = pd.read_csv('%s/minimal_abcd_clean.csv' % args.abcd_dir)
    p_guess_df = pd.read_csv('%s/p_guess_per_ssd.csv' % args.abcd_dir)

    SSD0_RTs = abcd_data.query(
        "SSDDur == 0.0 and correct_stop==0.0"
        ).stop_rt_adjusted.values
    sample_exgauss = generate_exgauss_sampler_from_fit(SSD0_RTs)

    simulator_dict = {
        'standard': SimulateData(),
        'guesses': SimulateData(guesses=True),
        'graded_mu_go_log': SimulateData(mu_go_grader='log'),
        'graded_mu_go_linear': SimulateData(mu_go_grader='linear')
    }

    group_data_dict = {
        'standard': pd.DataFrame(),
        'guesses': pd.DataFrame(),
        'graded_mu_go_log': pd.DataFrame(),
        'graded_mu_go_linear': pd.DataFrame(),
    }

    params = {
        'n_trials_stop': args.n_trials,
        'n_trials_go': args.n_trials,
        'SSDs': list(p_guess_df.columns.astype(float)),
        'guess_function': sample_exgauss,
        'p_guess_stop': list(p_guess_df.values.astype(float)[0]),
    }

    for sim_key in simulator_dict:
        print(sim_key)
        data = simulator_dict[sim_key].simulate(params)
        data['simulation'] = sim_key
        print('saving...')
        data.to_csv('%s/%s.csv' % (args.out_dir, sim_key))
