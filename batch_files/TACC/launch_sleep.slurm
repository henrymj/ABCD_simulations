#!/bin/bash
#
# SLURM control file automatically created by launch
#SBATCH -N 1
#SBATCH -o .out/launch.o%j
#SBATCH -p normal
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH --mail_type=ALL
#SBATCH --mail_user=henrymj@stanford.edu
# Job Submission
#-----------
module load launcher
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=sleep.sh
$LAUNCHER_DIR/paramrun
echo " "
echo " Parameteric Job Complete"
echo " "
