import json
import pandas as pd
import argparse
from simulate import generate_exgauss_sampler_from_fit

from utils import SimulateData


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulations')
    parser.add_argument('--n_trials', default=2000)
    parser.add_argument('--subjects', nargs='+',
                        help='subjects to run simulations on', required=True)
    parser.add_argument('--abcd_dir', default='./abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--out_dir',
                        default='./simulated_data/individual_data',
                        help='location to save simulated data')
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':
    print('getting args')
    args = get_args()
    print('analyzing ABCD info')
    # GET ABCD INFO
    p_guess_df = pd.read_csv('%s/p_guess_per_ssd.csv' % args.abcd_dir)
    p_guess_df.columns = p_guess_df.columns.astype(float)

    abcd_data = pd.read_csv('%s/minimal_abcd_clean.csv' % args.abcd_dir)
    SSD0_RTs = abcd_data.query(
        "SSDDur == 0.0 and correct_stop==0.0"
        ).stop_rt_adjusted.values
    sample_exgauss = generate_exgauss_sampler_from_fit(SSD0_RTs)

    indiv_ssd_dists = pd.read_csv('%s/SSD_dist_by_subj.csv' % args.abcd_dir,
                                  index_col=0)
    with open('%s/individual_mus.json' % args.abcd_dir) as json_file:
        mus_dict = json.load(json_file)

    # SETUP SIMULATORS

    simulator_dict = {
        'standard': SimulateData(),
        'guesses': SimulateData(guesses=True),
        'graded_mu_go_log': SimulateData(mu_go_grader='log'),
    }

    group_data_dict = {
        'standard': pd.DataFrame(),
        'guesses': pd.DataFrame(),
        'graded_mu_go_log': pd.DataFrame(),
    }

    params = {
        'n_trials_stop': args.n_trials,
        'n_trials_go': args.n_trials,
        'guess_function': sample_exgauss,
    }

    # SIMULATE INDIVIDUALS

    for sub in args.subjects:
        SSDs = indiv_ssd_dists.loc[sub, 'SSDDur'].unique()
        params['SSDs'] = SSDs
        params['p_guess_stop'] = list(p_guess_df[SSDs].values.astype(float)[0])

        params['mu_go'] = mus_dict[sub]['go']
        params['mu_stop'] = mus_dict[sub]['stop']

        for sim_key in simulator_dict:
            data = simulator_dict[sim_key].simulate(params)
            data['simulation'] = sim_key
            data.to_csv('%s/%s_%s.csv' % (args.out_dir, sim_key, str(sub)))
