import pandas as pd
import argparse
import json
from os import path
from glob import glob


from simulate import generate_exgauss_sampler_from_fit

from compute_SSRTs import generate_out_df,\
    simulate_graded_RTs_and_sort


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulations')
    parser.add_argument('--n_graded_go_trials', default=2000)
    parser.add_argument('--subjects', nargs='+',
                        help='subjects to run simulations on', required=True)
    parser.add_argument('--abcd_dir', default='./abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--sim_dir',
                        default='./simulated_data/individual_data',
                        help='location to save simulated data')
    parser.add_argument('--out_dir',
                        default='./ssrt_metrics/individual_metrics',
                        help='location to save simulated data')
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':
    args = get_args()

    # GET ABCD INFO
    abcd_data = pd.read_csv('%s/minimal_abcd_clean.csv' % args.abcd_dir)
    p_guess_df = pd.read_csv('%s/p_guess_per_ssd.csv' % args.abcd_dir)

    SSD_guess_dict = {float(col): float(p_guess_df[col].values[0]) for col
                      in p_guess_df.columns}

    indiv_ssd_dists = pd.read_csv('%s/SSD_dist_by_subj.csv' % args.abcd_dir,
                                  index_col=0)
    with open('%s/individual_mus.json' % args.abcd_dir) as json_file:
        mus_dict = json.load(json_file)

    SSD0_RTs = abcd_data.query(
        "SSDDur == 0.0 and correct_stop==0.0"
        ).stop_rt_adjusted.values
    sample_exgauss = generate_exgauss_sampler_from_fit(SSD0_RTs)

    # CALCULATE SSRT
    issue_subs = []
    for sub in args.subjects:
        try:
            params = {
                'mu_go': mus_dict[sub]['go'],
                'mu_stop': mus_dict[sub]['stop']
            }
            sub_SSDs = indiv_ssd_dists.loc[sub, 'SSDDur'].unique()
            graded_go_dict = {}
            for SSD in sub_SSDs:
                graded_go_dict[SSD] = simulate_graded_RTs_and_sort(
                    args.n_graded_go_trials,
                    SSD,
                    sub_params=params)

            for data_file in glob(path.join(args.sim_dir, '*%s*.csv' % sub)):
                sim_type = path.basename(
                    data_file
                    ).replace('.csv', '')
                out_df = generate_out_df(pd.read_csv(data_file),
                                        SSD_guess_dict,
                                        graded_go_dict,
                                        sample_exgauss)
                out_df.to_csv(path.join(args.out_dir, '%s.csv' % sim_type))
        except KeyError as err:
            print("KeyError error for sub {0}: {1}".format(sub, err))
            issue_subs.append(sub)
            continue
if len(issue_subs > 0):
    print('issue subs: ', issue_subs)
else:
    print('no problematic subs run here!')
