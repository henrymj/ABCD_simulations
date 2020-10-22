#!/bin/bash
#
# SLURM control file automatically created by launch
#SBATCH -N 1 # number of nodes requested - set to ceil(n rows in command script / 48)
#SBATCH -J launch       # Job Name
#SBATCH -o launch.o%j # Name of the output file )
#SBATCH -p normal
#SBATCH -t 02:30:00 # set to 2 hours - make sure that each script will finish in less than 2 hours
#SBATCH -n 1 # total number of mpi tasks requested - set to n rows in command script
#-----------
# Job Submission
#-----------
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=run_SSRTs.sh
$LAUNCHER_DIR/paramrun
echo " "
echo " Parameteric Job Complete"
echo " "
