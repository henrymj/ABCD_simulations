import numpy as np
import pandas as pd
import argparse
from os import path
from glob import glob

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


if __name__ == '__main__':
    args = get_args()

    # GET ABCD INFO
    abcd_data = pd.read_csv('%s/minimal_abcd_clean.csv' % args.abcd_dir)
    p_guess_df = pd.read_csv('%s/p_guess_per_ssd.csv' % args.abcd_dir)

    SSD_guess_dict = {float(col): float(p_guess_df[col].values[0]) for col
                      in p_guess_df.columns}

    SSD0_RTs = abcd_data.query(
        "SSDDur == 0.0 and correct_stop==0.0"
        ).stop_rt_adjusted.values
    sample_exgauss = generate_exgauss_sampler_from_fit(SSD0_RTs)

    # SET UP GRADED MU GO DISTS
    graded_go_dict = {}
    for SSD in [i for i in abcd_data.SSDDur.unique() if i == i]:
        graded_go_dict[SSD] = simulate_graded_RTs_and_sort(
            args.n_graded_go_trials,
            SSD)

    # CALCULATE SSRT
    for data_file in glob(path.join(args.sim_dir, '*.csv')):
        sim_type = path.basename(
            data_file
            ).replace('.csv', '')
        out_df = generate_out_df(pd.read_csv(data_file),
                                 SSD_guess_dict,
                                 graded_go_dict,
                                 sample_exgauss)
        out_df.to_csv(path.join(args.out_dir, '%s.csv' % sim_type))
