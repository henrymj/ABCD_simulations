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
    for sim_key in ['standard', 'guesses', 'graded_go', 'graded_both']:
        sub_dict[sim_key] = set(
            strip_paths_to_subs(
                glob(path.join(dir_path, '%s_*.csv' % sim_key)),
                sim_key))
    completed_subs = sub_dict['standard'].intersection(
        sub_dict['guesses'],
        sub_dict['graded_go'],
        sub_dict['graded_both'])
    return completed_subs


if __name__ == '__main__':

    # CONSTANTS
    SSRT_SCALES = [85, 25, 0]

    sher_header = '''#!/bin/bash
#SBATCH --job-name=ssrt
#SBATCH --output=.out/ssrt%d.out
#SBATCH --error=.err/ssrt%d.err
#SBATCH --time=3:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=henrymj@stanford.edu
#SBATCH -N 1 # number of nodes requested - set to ceil(n rows in command script / 24)
#SBATCH -n %d # total number of mpi tasks requested - set to n rows in command script
#SBATCH -p russpold,normal
# Job Submission
#-----------
export PYTHONPATH=""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py3-env

'''
    ABCD_LOC = '/oak/stanford/groups/russpold/users/henrymj/ABCD_simulations/abcd_data'
    SIM_LOC = '/oak/stanford/groups/russpold/users/henrymj/ABCD_simulations/simulated_data/individual_data'
    SSRT_LOC = '/oak/stanford/groups/russpold/users/henrymj/ABCD_simulations/ssrt_metrics/individual_metrics'

    # HELPER FUNCTIONS USING ABOVE PATHS (yes they should be pulled out I know)
    def get_mu_suffix(SSRT_SCALE=85, GORT_SCALE=93.84023748232624):
        stop_str_suffix = 'SSRTscale-%d' % SSRT_SCALE
        go_str_suffix = ''
        if GORT_SCALE != 93.84023748232624:
            go_str_suffix = '_RTscale-%d' % GORT_SCALE
        return '%s%s' % (stop_str_suffix, go_str_suffix)

    def get_remaining_subs(mu_suffix):
        simulated_subs = get_completed_subs(SIM_LOC+'_'+mu_suffix)
        subs_w_ssrts = get_completed_subs(SSRT_LOC+'_'+mu_suffix)
        remaining_subs = set(simulated_subs).difference(subs_w_ssrts)
        assert len(remaining_subs) == (len(simulated_subs) - len(subs_w_ssrts))
        return np.array(list(remaining_subs))

    def make_sherlock_ssrt_batch_files(sher_sim_file, suffix):
        nsubs_per_job = 72
        njobs_per_node = 24
        nlines = 0
        batch_counter = 0

        remaining_subs = get_remaining_subs(suffix)

        file_str = sher_header
        for start_idx in range(0, len(remaining_subs), nsubs_per_job):
            end_idx = start_idx + nsubs_per_job
            end_idx = min(end_idx, len(remaining_subs))
            substr = ' '.join(remaining_subs[start_idx:end_idx])
            file_str += (f'eval "python ../../scripts/compute_indiv_SSRTs.py --mu_suffix {suffix} --abcd_dir {ABCD_LOC} --sim_dir_base {SIM_LOC} --out_dir_base {SSRT_LOC} --subjects {substr}" &\n')
            nlines += 1
            if nlines == njobs_per_node:
                with open(sher_sim_file % batch_counter, 'w') as f:
                    f.write(file_str % (batch_counter, batch_counter, nlines))
                    f.write(f'wait\n')
                # reset for new batch file
                file_str = sher_header
                nlines = 0
                batch_counter += 1
        # at end, if things didn't split out evenly, write out the remaining subs
        if nlines > 0:
            with open(sher_sim_file % batch_counter, 'w') as f:
                f.write(file_str % (batch_counter, batch_counter, nlines))
                f.write(f'wait\n')

    # MAIN BODY
    for SSRT_SCALE in SSRT_SCALES:
        suffix = get_mu_suffix(SSRT_SCALE)
        sher_file = 'batch_files/sherlock/%s/' % suffix + '/sherlock_run_ssrt_iter%d.batch'
        make_sherlock_ssrt_batch_files(sher_file, suffix)

    # Single Individual Case
    suffix_noVar = get_mu_suffix(0, 0)
    sher_file = 'batch_files/sherlock/%s/' % suffix_noVar + '/sherlock_run_ssrt_iter%d.batch'
    make_sherlock_ssrt_batch_files(sher_file, suffix_noVar)
