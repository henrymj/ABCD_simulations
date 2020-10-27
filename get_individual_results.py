#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sstats
from glob import glob
import numpy as np
from generate_SSRT_cmds import get_completed_subs
import feather
import os
from dask import delayed
import dask.dataframe as dd
import seaborn as sns


# In[2]:


subs_w_ssrts = get_completed_subs('ssrt_metrics/individual_metrics')
subs_w_ssrts = list(subs_w_ssrts)


# In[3]:


ssrt_metrics = dd.read_csv('ssrt_metrics/individual_metrics/*.csv', include_path_column='filename')
ssrt_metrics['NARGUID'] = ssrt_metrics['filename'].apply(lambda x: x.split('_')[-1].replace('.csv', ''), meta=str)
ssrt_metrics['underlying distribution'] = ssrt_metrics['filename'].apply(lambda x: x.split('/')[-1].split('_')[0], meta=str)
ssrt_metrics['underlying distribution'] = ssrt_metrics['underlying distribution'].map({'graded': 'graded_mu_go_log',
                                                                                       'standard': 'standard',
                                                                                       'guesses': 'guesses'})
ssrt_metrics = ssrt_metrics.drop('filename', axis=1)
ssrt_metrics = ssrt_metrics.rename(columns={'SSRT': 'standard',
                                            'SSRT_w_guesses': 'guesses',
                                            'SSRT_w_graded': 'graded_mu_go_log'})


# In[4]:


melt_df = dd.melt(ssrt_metrics, id_vars=['SSD', 'underlying distribution'], value_vars=['standard', 'guesses', 'graded_mu_go_log'], var_name = 'assumed distribution', value_name='SSRT')


# In[5]:


fig,ax = plt.subplots(1,1, figsize=(14, 8))
_ = sns.lineplot(x='SSD', y='SSRT', color='k', style='underlying distribution', data=melt_df[(melt_df['assumed distribution'] == 'standard') & (melt_df['SSD'] <= 650)].compute(), linewidth=3)
plt.savefig('figures/SSRT_by_SSD.png')


# In[6]:


fig,ax = plt.subplots(1,1, figsize=(14, 8))
keep_idx = ((melt_df['assumed distribution'] == 'standard') | (melt_df['assumed distribution'] == melt_df['underlying distribution'])) & (melt_df['SSD'] <= 650)
_ = sns.lineplot(x='SSD', y='SSRT', hue='assumed distribution', style='underlying distribution', data=melt_df[keep_idx].compute(), palette=['k', '#1f77b4', '#ff7f0e'], linewidth=3)
plt.savefig('figures/SSRT_by_SSD_supplement.png')


# In[7]:


abcd_inhib_func_per_sub = dd.read_csv('abcd_data/abcd_inhib_func_per_sub.csv')
full_inhib_func_df = dd.concat([ssrt_metrics[abcd_inhib_func_per_sub.columns], abcd_inhib_func_per_sub], 0)
fig,ax = plt.subplots(1,1, figsize=(14, 8))
_ = sns.lineplot(x='SSD', y='p_respond', color='k', style='underlying distribution', data=full_inhib_func_df.query('SSD <= 500').compute(), linewidth=3)
_ = plt.ylim([0,1])
plt.savefig('figures/inhibition_function.png')


# # Indvidual Differences

# In[8]:


ABCD_SSD_dists = pd.read_csv('abcd_data/SSD_dist_by_subj.csv')


# In[9]:


def weight_ssrts(sub_df):
    sub_df = sub_df.copy()
    indiv_SSRT = np.zeros((1,3)) # pd.DataFrame([0,0,0], columns=['standard', 'guesses', 'graded_mu_go_log'])
    sub = sub_df['NARGUID'].unique()[0]
    sub_dists = ABCD_SSD_dists.query("NARGUID=='%s'" % sub)
#     gen_str = sub_df['underlying distribution'].unique()[0]
    for SSD in sub_dists.SSDDur:
        ssd_SSRTs = sub_df.loc[sub_df.SSD==SSD, ['standard', 'guesses', 'graded_mu_go_log']].values[0]
        weight = sub_dists.loc[sub_dists.SSDDur==SSD, 'proportion'].values
        indiv_SSRT += ssd_SSRTs * weight
    return pd.DataFrame(indiv_SSRT, columns=['standard', 'guesses', 'graded_mu_go_log'])


# In[10]:


expected_ssrts = ssrt_metrics.groupby(['NARGUID', 'underlying distribution']).apply(lambda x: weight_ssrts(x))


# In[11]:


expected_ssrts = expected_ssrts.compute()
expected_ssrts = expected_ssrts.reset_index()
expected_ssrts


# In[14]:


pivot_ssrts = expected_ssrts.pivot(index='NARGUID', columns='underlying distribution', values=['standard', 'guesses', 'graded_mu_go_log'])


# In[15]:


pivot_ssrts.corr(method='spearman')


# In[16]:


pivot_ssrts.to_csv('ssrt_metrics/expected_ssrts.csv')

