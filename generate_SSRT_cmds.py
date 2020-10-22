#!/usr/bin/env python
# coding: utf-8

# In[22]:
import numpy as np
from glob import glob
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove, path


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


def get_completed_subs(dir_path):
    sub_dict = {}
    for sim_key in ['standard', 'guesses', 'graded_mu_go_log']:
        sub_dict[sim_key] = set(
            strip_paths_to_subs(
                glob(path.join(dir_path, '%s_*.csv' % sim_key)),
                sim_key))
    completed_subs = sub_dict['standard'].intersection(
        sub_dict['guesses'],
        sub_dict['guesses'])
    return completed_subs


if __name__ == '__main__':
    # ## get subs w SSRT computations
    # In[16]:
    subs_w_ssrts = get_completed_subs('ssrt_metrics/individual_metrics')

    # ## get simulated subs
    # In[14]:

    simulated_subs = get_completed_subs('simulated_data/individual_data')

    # In[15]:

    # ## get remainder and write script sh file
    # In[28]:
    remaining_subs = set(simulated_subs).difference(subs_w_ssrts)
    assert len(remaining_subs) == (len(simulated_subs) - len(subs_w_ssrts))

    remaining_subs = np.array(list(remaining_subs))

    # In[31]:
    nsubs_per_job = 48
    nlines = 0
    with open('run_SSRTs.sh', 'w') as f:
        for start_idx in range(0, len(remaining_subs), nsubs_per_job):
            end_idx = start_idx + nsubs_per_job
            if end_idx > len(remaining_subs):
                end_idx = len(remaining_subs)
            substr = ' '.join(remaining_subs[start_idx:end_idx])
            f.write(f'python compute_indiv_SSRTs.py --subjects {substr}\n')
            nlines += 1

    N_line_str = '#SBATCH -N %d # number of nodes requested - set to ceil(n rows in command script / 48)\n' % int(np.ceil(nlines/nsubs_per_job))
    n_line_str = '#SBATCH -n %s # total number of mpi tasks requested - set to n rows in command script\n' % nlines

    replace(
        'launch_SSRT_cmds.slurm',
        '#SBATCH -N',
        N_line_str)
    replace(
        'launch_SSRT_cmds.slurm',
        '#SBATCH -n',
        n_line_str)
    # prints so you can compare the lines of the slurm file
    print(N_line_str)
    print(n_line_str)
