#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import dask.dataframe as dd
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='ABCD sim results')
    parser.add_argument('--job',
                        default='all',
                        help='choose one from [plot_ssrts,\
                              plot_inhib_func, calc_ssrts, all]',
                        type=str)
    parser.add_argument('--abcd_dir', default='../abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--ssrt_dir',
                        default='../ssrt_metrics',
                        help='location to save simulated data')
    parser.add_argument('--fig_dir',
                        default='../figures',
                        help='location to save simulated data')
    args = parser.parse_args()
    return(args)


def weight_ssrts(sub_df, ABCD_SSD_dists):
    sub_df = sub_df.copy()
    indiv_SSRT = np.zeros((1, 4))
    sub = sub_df['NARGUID'].unique()[0]
    sub_dists = ABCD_SSD_dists.query("NARGUID=='%s'" % sub)
    for SSD in sub_dists.SSDDur:
        ssd_SSRTs = sub_df.loc[sub_df.SSD == SSD,
                               ['standard', 'guesses', 'graded_go', 'graded_both']
                               ].values[0]
        weight = sub_dists.loc[sub_dists.SSDDur == SSD, 'proportion'].values
        indiv_SSRT += ssd_SSRTs * weight
    return pd.DataFrame(indiv_SSRT,
                        columns=['standard', 'guesses', 'graded_go', 'graded_both'])


# In[3]:
if __name__ == '__main__':
    print('getting args...')
    args = get_args()
    print('job = %s' % args.job)

    print('loading in data...')
    ssrt_metrics = dd.read_csv('%s/individual_metrics/*.csv' % args.ssrt_dir,
                               include_path_column='filename')
    ssrt_metrics['NARGUID'] = ssrt_metrics['filename'].apply(
        lambda x: x.split('_')[-1].replace('.csv', ''), meta=str)
    ssrt_metrics['underlying distribution'] = ssrt_metrics['filename'].apply(
        lambda x: '_'.join(x.split('/')[-1].split('_')[:-1]), meta=str)
    # ssrt_metrics['underlying distribution'] = ssrt_metrics[
    #     'underlying distribution'
    #     ].map({'graded': 'graded_mu_go',
    #            'standard': 'standard',
    #            'guesses': 'guesses'})
    ssrt_metrics = ssrt_metrics.drop('filename', axis=1)
    ssrt_metrics['graded_both'] = ssrt_metrics['SSRT_w_graded']
    ssrt_metrics = ssrt_metrics.rename(
        columns={'SSRT': 'standard',
                 'SSRT_w_guesses': 'guesses',
                 'SSRT_w_graded': 'graded_go'})

    print('melting...')
    melt_df = dd.melt(ssrt_metrics,
                      id_vars=['SSD', 'underlying distribution'],
                      value_vars=['standard', 'guesses', 'graded_go', 'graded_both'],
                      var_name='assumed distribution',
                      value_name='SSRT')

    if args.job in ['plot_ssrts', 'all']:
        print('plotting SSRT by SSD Supplement...')
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        keep_idx = (
            (melt_df['assumed distribution'] == 'standard') |
            (melt_df['assumed distribution'] == melt_df['underlying distribution'])
            ) &\
            (melt_df['SSD'] <= 650)
        subset_melt_df = melt_df[keep_idx].compute()
        _ = sns.lineplot(x='SSD',
                         y='SSRT',
                         hue='assumed distribution',
                         style='underlying distribution',
                         data=subset_melt_df,
                         palette=['k', '#1f77b4', '#ff7f0e', '#2ca02c'],
                         linewidth=3)
        plt.savefig('%s/SSRT_by_SSD_supplement.png' % args.fig_dir)

        print('plotting SSRT by SSD...')
        fig_idx = (subset_melt_df['assumed distribution'] == 'standard') &\
                  (subset_melt_df['SSD'] <= 650)
        main_fix_melt_df = subset_melt_df[fig_idx]
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        _ = sns.lineplot(
            x='SSD',
            y='SSRT',
            color='k',
            style='underlying distribution',
            data=main_fix_melt_df,
            linewidth=3)
        plt.savefig('%s/SSRT_by_SSD.png' % args.fig_dir)

    if args.job in ['plot_inhib_func', 'all']:
        print('plotting Inhibition Function...')
        abcd_inhib_func_per_sub = dd.read_csv(
            '%s/abcd_inhib_func_per_sub.csv' % args.abcd_dir)
        full_inhib_func_df = dd.concat(
            [ssrt_metrics[abcd_inhib_func_per_sub.columns],
             abcd_inhib_func_per_sub],
            0)
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        _ = sns.lineplot(x='SSD',
                         y='p_respond',
                         color='k',
                         style='underlying distribution',
                         data=full_inhib_func_df.query('SSD <= 500').compute(),
                         linewidth=3)
        _ = plt.ylim([0, 1])
        plt.savefig('%s/inhibition_function.png' % args.fig_dir)

    if args.job in ['calc_ssrts', 'all']:
        print('Calculating Expected SSRTs...')
        ABCD_SSD_dists = pd.read_csv('%s/SSD_dist_by_subj.csv' % args.abcd_dir)

        expected_ssrts = ssrt_metrics.groupby(
            ['NARGUID', 'underlying distribution']
            ).apply(lambda x: weight_ssrts(x, ABCD_SSD_dists))

        print('Running the Dask Computation...')
        expected_ssrts = expected_ssrts.compute()
        expected_ssrts = expected_ssrts.reset_index()
        expected_ssrts

        pivot_ssrts = expected_ssrts.pivot(
            index='NARGUID',
            columns='underlying distribution',
            values=['standard', 'guesses', 'graded_mu_go_log']
            )

        print('Saving expected SSRTs')
        pivot_ssrts.to_csv('%s/expected_ssrts.csv' % args.ssrt_dir)
