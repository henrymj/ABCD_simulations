# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# pyABC example

import pyabc
import json
from pyabc import (ABCSMC,
                   RV, Distribution)
import numpy as np
import scipy.stats as st
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt

from ssd import fixedSSD
from stoptaskstudy import StopTaskStudy
# %matplotlib inline


# +
def stopsignal_model(parameters):
    paramfile = 'params.json'
    with open(paramfile) as f:
            params = json.load(f)
    # install the parameters from the simulation
    params['mu']['go'] = parameters['mu_go']
    params['mu']['stop'] = parameters['mu_go'] + parameters['mu_stop_delta']
    params['mu_delta_incorrect']  = parameters['mu_delta_incorrect']
    # TBD
    #    if args.p_guess_file is not None:
    #        p_guess = pd.read_csv(args.p_guess_file, index_col=0)
    #        assert 'SSD' in p_guess.columns and 'p_guess' in p_guess.columns

    #    if args.random_seed is not None:
    #        np.random.seed(args.random_seed)

    min_ssd, max_ssd, ssd_step = 0, 550, 50
    ssd = fixedSSD(np.arange(min_ssd, max_ssd + ssd_step, ssd_step))

    study = StopTaskStudy(ssd, None, params=params)

    trialdata = study.run()
    metrics = study.get_stopsignal_metrics()
    # summarize data - go trials are labeled with SSD of -inf so that
    # they get included in the summary
    stop_data = trialdata.groupby('SSD').mean().query('SSD >= 0').resp.values
    return {'presp': stop_data,
            'mean_go_RT': metrics['mean_go_RT'],
            'mean_stopfail_RT': metrics['mean_stopfail_RT']}

parameter_prior = Distribution(mu_go=RV("uniform", 0, 1),
                               mu_stop_delta=RV("uniform", 0, 1),
                              mu_delta_incorrect=RV("uniform", 0, 1))
parameter_prior.get_parameter_names()


# +


def rmse(a, b):
    return(np.sqrt(np.sum((a - b)**2)))

def distance(simulation, data):
    presp_rmse = rmse(simulation['presp'], data['presp'])
    gort_rmse = rmse(simulation['mean_go_RT'], data['go_rt'])/200
    stopfailrt_rmse = rmse(simulation['mean_stopfail_RT'], data['stopfail_rt'])/10
    return(presp_rmse + gort_rmse + stopfailrt_rmse)

abc = ABCSMC(stopsignal_model, parameter_prior, distance)

# -

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))
observed_presp = pd.read_csv('presp_by_ssd_inperson.txt',  delimiter=r"\s+", index_col=0)
abc.new(db_path, {"presp": observed_presp.presp.values, 'go_rt': 455.367, 'stopfail_rt': 219.364})

history = abc.run(minimum_epsilon=.1, max_nr_populations=10)

fig, ax = plt.subplots()
for t in range(history.max_t+1):
    df, w = history.get_distribution(m=0, t=t)
    pyabc.visualization.plot_kde_1d(
        df, w,
        xmin=0, xmax=1,
        x="mu_go", ax=ax,
        label="PDF t={}".format(t))
#ax.axvline(observed_presp.presp.values, color="k", linestyle="dashed");
ax.legend();


df, w = history.get_distribution(m=0, t=history.max_t)
plt.scatter(df, w)

df.iloc[np.argmin(w),:]

presp = stopsignal_model({'mu_go':0.774443, 'mu_stop_delta': 0.129610, 'mu_delta_incorrect': 0.226288})

plt.plot(presp['data'])
plt.plot(observed_presp.presp.values, 'k')


