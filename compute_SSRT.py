import pandas as pd
import argparse

from stopsignalmetrics import SSRTmodel


def get_args():
    parser = argparse.ArgumentParser(description='simulation visualization')
    parser.add_argument("--sim_types",
                        nargs="+",
                        default=['vanilla', 'guesses', 'log', 'linear'])
    parser.add_argument('--sim_dir',
                        default='./simulated_data',
                        help='location of simulated data')
    parser.add_argument('--out_dir',
                        default='./ssrt_metrics',
                        help='location to save SSRT metrics')
    parser.add_argument('--ssrt_model',
                        default='all')
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':

    args = get_args()
    ssrtmodel = SSRTmodel(model=args.ssrt_model)
    for sim in args.sim_types:
        sim_df = pd.read_csv('%s/%s.csv' % (args.sim_dir, sim))
        metrics_df = ssrtmodel.fit_transform(sim_df, level='group')
        metrics_df.to_csv('%s/%s.csv' % (args.out_dir, sim))
