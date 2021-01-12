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
    parser.add_argument('--mu_suffix', required=True, type=str)
    parser.add_argument('--abcd_dir', default='../abcd_data',
                        help='location of ABCD data')
    parser.add_argument('--ssrt_dir',
                        default='../ssrt_metrics',
                        help='location of ssrt metrics')
    parser.add_argument('--fig_dir',
                        default='../figures',
                        help='location to save figures')
    parser.add_argument('--clip_SSDs_bool',
                        default=True,
                        help='clip fixed SSD design to clipped_SSD instead of max',
                        type=bool)
    parser.add_argument('--clipped_SSD',
                        default=500,
                        help='max SSD to use if dist is clipped')
    return(parser.parse_args())


def weight_ssrts(sub_df, ABCD_SSD_dists):
    sub_df = sub_df.copy()
    indiv_SSRT = np.zeros((1, 6))
    sub = sub_df['NARGUID'].unique()[0]
    sub_dists = ABCD_SSD_dists.query("NARGUID=='%s'" % sub)
    for SSD in sub_dists.SSDDur:
        ssd_SSRTs = sub_df.loc[sub_df.SSD == SSD,
                               ['standard', 'guesses', 'graded_go', 'graded_both']
                               ].values[0]
        weight = sub_dists.loc[sub_dists.SSDDur == SSD, 'proportion'].values
        indiv_SSRT[0][:4] += ssd_SSRTs * weight
    # append fixed, tracked
    indiv_SSRT[0][4] = sub_df.loc[sub_df.SSD == -np.inf, 'standard'].values[0]
    try:
        indiv_SSRT[0][5] = sub_df.loc[sub_df.SSD == np.inf, 'standard'].values[0]
    except IndexError as err:
        print(
            "Index Error for sub '{0}', underlying gen '{1}', len {2}: {3}".format(
                sub,
                sub_df['underlying distribution'].unique()[0],
                len(sub_df),
                err
                )
            )
        print(sub_df.loc[sub_df.SSD == np.inf, :])
        print('setting to NaN')
        indiv_SSRT[0][5] = np.nan

    return pd.DataFrame(indiv_SSRT,
                        columns=['standard',
                                 'guesses',
                                 'graded_go',
                                 'graded_both',
                                 'fixed',
                                 'tracking'])


# In[3]:
if __name__ == '__main__':
    print('getting args...')
    args = get_args()
    print('job = %s' % args.job)

    print('loading in data...')
    ssrt_metrics = dd.read_csv('%s/individual_metrics_%s/*.csv' % (args.ssrt_dir, args.mu_suffix),
                               include_path_column='filename',
                               dtype={'Unnamed: 0': 'float64',
                                      'omission_count': 'float64'})
    ssrt_metrics['NARGUID'] = ssrt_metrics['filename'].apply(
        lambda x: x.split('_')[-1].replace('.csv', ''), meta=str)
    ssrt_metrics['underlying distribution'] = ssrt_metrics['filename'].apply(
        lambda x: '_'.join(x.split('/')[-1].split('_')[:-1]), meta=str)
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
        plt.savefig('%s/%s/SSRT_by_SSD_supplement.png' % (args.fig_dir, args.mu_suffix), dpi=600)

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
        plt.savefig('%s/%s/SSRT_by_SSD.png' % (args.fig_dir, args.mu_suffix), dpi=600)

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
        plt.savefig('%s/%s/inhibition_function.png' % (args.fig_dir, args.mu_suffix), dpi=600)

    if args.job in ['calc_ssrts', 'all']:
        print('Calculating Expected SSRTs...')

        # SSD distributions for individuals
        ssd_dist_path = '%s/SSD_dist_by_subj_%sClip-%s.csv' % (args.abcd_dir,
                                                               args.clipped_SSD,
                                                               args.clip_SSDs_bool)
        ABCD_SSD_dists = pd.read_csv(ssd_dist_path,
                                     index_col=0)

        print('Running the Dask Computation...')
        expected_ssrts = ssrt_metrics.compute().groupby(
            ['NARGUID', 'underlying distribution']
            ).apply(lambda x: weight_ssrts(x, ABCD_SSD_dists))

        expected_ssrts = expected_ssrts.reset_index()
        expected_ssrts

        pivot_ssrts = expected_ssrts.pivot(
            index='NARGUID',
            columns='underlying distribution',
            values=['standard', 'guesses', 'graded_go', 'graded_both', 'fixed', 'tracking']
            )

        print('Saving expected SSRTs')
        pivot_ssrts.to_csv('%s/expected_ssrts_%s.csv' % (args.ssrt_dir, args.mu_suffix))
