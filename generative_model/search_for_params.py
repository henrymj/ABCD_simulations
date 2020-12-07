# search for best params for non-abcd stopping data



import argparse
import os
import json
import pandas as pd
import numpy as np
from stoptaskstudy import fixedSSD, StopTaskStudy


def get_args():
    # defaults based on in-person data from henry
    parser = argparse.ArgumentParser(description='ABCD data search')
    parser.add_argument('--paramfile', help='json file containing starting parameters')
    parser.add_argument('--min_ssd', help='minimum SSD value', default=0)
    parser.add_argument('--max_ssd', help='maximum SSD value', default=550)
    parser.add_argument('--ssd_step', help='SSD step size', default=50)
    parser.add_argument('--random_seed', help='random seed', type=int)
    parser.add_argument('--tracking', help='use tracking algorithm', action='store_true')
    parser.add_argument('--n_subjects', type=int,
                        help='number of subjects to simulate', default=1)
    parser.add_argument('--out_dir',
                        default='./simulated_data/search',
                        help='location to save simulated data')
    parser.add_argument('--target_presp_file',
                        default='presp_by_ssd_inperson.txt',
                        help='file with target presp by SSD')
    parser.add_argument('--target_mean_go_rt', default=455.3)
    parser.add_argument('--target_mean_stopfail_rt', default=219.3)
    parser.add_argument('--target_mean_go_acc', default=.935)

    return parser.parse_args()

def score_results(metrics, params):
    pass


def rmse(a, b):
    return(np.sqrt(np.sum((a - b)**2)))


if __name__ == '__main__':
    args = get_args()
    print('searching for best parameters')
    if args.paramfile is not None:
        with open(args.paramfile) as f:
            params = json.load(f)
    else:
        params = None
    print(params)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    # load target data
    target_presp_by_ssd = pd.read_csv(args.target_presp_file, delimiter=r"\s+", index_col=0)

    ssd = fixedSSD(np.arange(args.min_ssd, args.max_ssd + args.ssd_step, args.ssd_step))

    mu_go_range = np.arange(0.3, 1.1, 0.05)
    mu_stop_delta_range = np.arange(0.05, 0.41, 0.05)
    mu_delta_incorrect_range = np.arange(0.05, 0.51, 0.05)
    paramsets = np.array(
        np.meshgrid(mu_go_range, 
                    mu_stop_delta_range,
                    mu_delta_incorrect_range)).T.reshape(-1,3)

    results = []
    for i in range(paramsets.shape[0]):

        print(f'running subject {i + 1}: {i/paramsets.shape[0]} complete')
        sim_params = params.copy()
        sim_params['mu'] = {'go': paramsets[i, 0],
                            'stop': paramsets[i, 0] + paramsets[i, 1]}
        sim_params['mu_delta_incorrect'] = paramsets[i, 2]
        study = StopTaskStudy(ssd, args.out_dir, params=sim_params)
        # save some extra params for output to json
        study.params['args'] = args.__dict__
        trialdata = study.run()
        study.save_trialdata()

        stop_data = trialdata.groupby('SSD').mean().query('SSD >= 0')

        metrics = study.get_stopsignal_metrics()
        study.save_metrics()

        # score metrics
        presp_rmse = rmse(stop_data.resp, target_presp_by_ssd.presp)
        gort_error = metrics['mean_go_RT'] - args.target_mean_go_rt
        stopfailrt_error = metrics['mean_stopfail_RT'] - args.target_mean_stopfail_rt
        goacc_error = metrics['go_acc'] - args.target_mean_go_acc

        results.append(paramsets[i, :].tolist() + [presp_rmse,  goacc_error, gort_error,stopfailrt_error])
        lksjdf
        
results_df = pd.DataFrame(results, columns=['mu_go', 'mu_stop_delta', 'mu_delta_incorrect',
                    'presp_rmse',  'goacc_error', 'gort_error','stopfailrt_error'])
results_df.to_csv('simulation_results.csv')
# %%
