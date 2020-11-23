#!/usr/bin/env python
# coding: utf-8

# In[22]:
import json
import pandas as pd
import numpy as np
from glob import glob
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove


def strip_paths_to_subs(filelist, key):
    return [f.split(key+'_')[-1].replace('.csv', '') for f in filelist]


def replace(file_path, pattern, subst):
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if pattern in line:
                    new_file.write(subst)
                else:
                    new_file.write(line)
    # Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    # Remove original file
    remove(file_path)
    # Move new file
    move(abs_path, file_path)


if __name__ == '__main__':
    # ## get all subs
    # In[16]:
    abcd_data = pd.read_csv('abcd_data/minimal_abcd_clean.csv')
    all_subs = abcd_data.NARGUID.unique()

    with open('abcd_data/individual_mus.json') as json_file:
        mus_dict = json.load(json_file)

    all_subs_filtered = set(all_subs).difference(set(mus_dict['prob_subs']))
    assert len(all_subs_filtered) == (len(all_subs) - len(mus_dict['prob_subs']))
    # ## get finished subs
    # In[14]:

    finished_sub_dict = {}
    for sim_key in ['standard', 'guesses', 'graded_mu_go_log']:
        finished_sub_dict[sim_key] = set(
            strip_paths_to_subs(
                glob('simulated_data/individual_data/%s_*.csv' % sim_key),
                sim_key))

    # In[15]:
    finished_subs = finished_sub_dict['standard'].intersection(
        finished_sub_dict['guesses'],
        finished_sub_dict['graded_mu_go_log'])

    # ## get remainder and write script sh file
    # In[28]:
    remaining_subs = set(all_subs_filtered).difference(finished_subs)
    assert len(remaining_subs) == (len(all_subs_filtered) - len(finished_subs))

    remaining_subs = np.array(list(remaining_subs))

    # In[31]:
    nsubs_per_job = 48
    njobs_per_node = 36
    nlines = 0
    with open('run_sims.sh', 'w') as f:
        for start_idx in range(0, len(remaining_subs), nsubs_per_job):
            end_idx = start_idx + nsubs_per_job
            if end_idx > len(remaining_subs):
                end_idx = len(remaining_subs)
            substr = ' '.join(remaining_subs[start_idx:end_idx])
            f.write(f'python simulate_individuals.py --subjects {substr}\n')
            nlines += 1

    N_line_str = '#SBATCH -N %d # number of nodes requested - set to ceil(n rows in command script / 48)\n' % int(np.ceil(nlines/njobs_per_node))
    n_line_str = '#SBATCH -n %s # total number of mpi tasks requested - set to n rows in command script\n' % nlines

    replace(
        'launch_sim_cmds.slurm',
        '#SBATCH -N',
        N_line_str)
    replace(
        'launch_sim_cmds.slurm',
        '#SBATCH -n',
        n_line_str)
    # prints so you can compare the lines of the slurm file
    print(N_line_str)
    print(n_line_str)
