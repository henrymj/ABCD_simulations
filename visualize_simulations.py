import numpy as np
import pandas as pd
import argparse
from glob import glob
from os import path
from utils import joyplot

import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib
matplotlib.use('Agg')


def get_args():
    parser = argparse.ArgumentParser(description='simulation visualization')
    parser.add_argument('--sim_dir',
                        default='./simulated_data',
                        help='location of simulated data')
    parser.add_argument('--out_dir',
                        default='./visualizations',
                        help='location to save SSRT metrics')
    args = parser.parse_args()
    return(args)


def plot_RTs_per_SSD(sim_data, args, filename=''):

    SSDs = sim_data.SSD.unique()
    SSDs = [i for i in SSDs if i == i]
    SSDs.sort()
    RT_dist_dict = {ssd: [] for ssd in SSDs}
    for ssd in SSDs:
        RT_dist_dict[ssd] = sim_data.query(
            f"SSD == {ssd}"
        ).stopRT.values

    go_RTs = sim_data.query(
            f"condition == 'go'"
        ).goRT.values

    RT_dist_dict.update({-1: go_RTs})

    max_RT = np.nanmax([np.nanmax(RT_dist_dict[key]) for key in RT_dist_dict])

    bins = np.arange(0, max_RT, 5)

    RT_by_SSD_df = pd.DataFrame({key: pd.Series(value) for key, value
                                in RT_dist_dict.items()})
    RT_by_SSD_df = RT_by_SSD_df.reindex(sorted(RT_by_SSD_df.columns), axis=1)
    rt_by_SSD_melt = RT_by_SSD_df.melt(
        value_vars=RT_by_SSD_df.columns,
        var_name='SSD',
        value_name='RT')

    fig, axes = joyplot(rt_by_SSD_melt, by="SSD", column="RT",
                        range_style='own',
                        grid='y',
                        linewidth=1, legend=False, figsize=(8, 8),
                        title="RT Distribution by trial type / SSD",
                        bins=bins,
                        hist=True,
                        density=True,
                        ylim='own',
                        colormap=cm.autumn_r,
                        fade=True)

    plt.savefig('%s/%s.png' % (args.out_dir, filename))
    plt.close()


if __name__ == '__main__':

    args = get_args()
        
    for sim_file in glob(path.join(args.sim_dir, '*')):
        sim_key = sim_file.split('/')[-1].replace('.csv', '')
        sim_df = pd.read_csv(sim_file)
        plot_RTs_per_SSD(sim_df, args, filename=sim_key)

