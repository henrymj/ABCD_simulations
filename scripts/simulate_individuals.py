import json
import pandas as pd
import argparse
from utils import SimulateData


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


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulations')
    parser.add_argument('--n_trials', default=2000)
    parser.add_argument('--subjects', nargs='+',
                        help='subjects to run simulations on', required=True)
    parser.add_argument('--abcd_dir', default='../abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--out_dir',
                        default='../simulated_data/individual_data',
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
    with open('%s/assigned_mus.json' % args.abcd_dir) as json_file:
        mus_dict = json.load(json_file)

    # SETUP SIMULATORS

    simulator_dict = {
        'standard': SimulateData(),
        'guesses': SimulateData(guesses=True),
        'graded_go': SimulateData(grade_mu_go=True),
        'graded_both': SimulateData(grade_mu_go=True, grade_mu_stop=True),
    }

    params = {
        'n_trials_stop': args.n_trials,
        'n_trials_go': args.n_trials,
        'guess_function': sample_exgauss,
    }

    # SIMULATE INDIVIDUALS
    issue_subs = []
    for sub in args.subjects:
        try:
            SSDs = indiv_ssd_dists.loc[sub, 'SSDDur'].unique()
            params['SSDs'] = SSDs
            params['p_guess_stop'] = list(p_guess_df[SSDs].values.astype(float)[0])

            params['mu_go'] = mus_dict[sub]['go']
            params['mu_stop'] = mus_dict[sub]['stop']

            for sim_key in simulator_dict:
                data = simulator_dict[sim_key].simulate(params)
                data['simulation'] = sim_key
                data.to_csv('%s/%s_%s.csv' % (args.out_dir, sim_key, str(sub)))
        except KeyError as err:
            print("KeyError error for sub {0}: {1}".format(sub, err))
            issue_subs.append(sub)
            continue
if len(issue_subs) > 0:
    print('issue subs: ', issue_subs)
else:
    print('no problematic subs run here!')
