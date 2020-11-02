#!/bin/bash
#
# SLURM control file automatically created by launch
#SBATCH -N 3
#SBATCH -o .out/launch.o%j
#SBATCH -p normal
#SBATCH -t 02:30:00
#SBATCH -n 3
# Job Submission
#-----------
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=run_results.sh
$LAUNCHER_DIR/paramrun
echo " "
echo " Parameteric Job Complete"
echo " "
