# Investigating violations of the assumption of context independence in the ABCD dataset.

## Overview:

Coming soon.

## Running on all ABCD individuals (new simulations):

0. The most important files are: `simulate_individuals.py`, `compute_indiv_SSRTs.py`, `get_individual_results.py`, and `utils.py`.

1. `preprocess_ABCD_get_metrics.ipynb` drops participants affected by issue 3 and adds a choice accuracy column as preprocessing. It then computes a) SSRT per participant in order to get mu_go and mu_stop from their mean RT and SSRT, b) the SSD distributions for each participant, c) the accuracy and subsequent P(guess) for each SSD, and d) the inhibition function of the dataset. Each of these is saved to abcd_data/, but this is hidden from github.
Following this, it generates `run_sims.sh` and edits `launch_sim_cmds.slurm` for running n jobs in parallel on TACC. It now also makes `sherlock_run_sims.batch` in an attempt to run this these same jobs in parallel on Sherlock. 
  
2. run `sbatch launch_sim_cmds.slurm` or `sbatch sherlock_run_sims.batch` depending on your HPC. This will run `simulate_individuals.py`. 
- After the simulations "finish", run `python generate_remaining_sim_cmds.py` to look for subs with all 3 simulation files saved and updates `run_sims.sh` to simulate whichever subs didn't complete.

3. run `generate_SSRT_cmds.py` to compare simulated subs to subs with ssrt files and update `run_SSRTs.sh` with a list of subjects in need of ssrt files. run `launch_SSRT_cmds.slurm` which will read `run_SSRTs.sh` and run `compute_indiv_SSRTs.py`, computing SSRTs and getting related metrics per SSD per simulation type x SSRT assumption. 
- Rerunning `generate_SSRT_cmds.py` will update `run_SSRTs.sh` to get SSRT for any subjects which didn't complete.

4. run `launch_run_results.slurm` or `sherlock_run_results.batch` to run `get_individual_results.py`. This reads in the individual ssrt files generated in step 3 and creates the figures along with the expected SSRT for each simulation type x SSRT assumption (9 expected SSRTs) per subject.

5. run `display_results.ipynb` to view the figures and examine the rank correlations across the simulation type x SSRT assumption combination.

## Running single individual (old simulations):

1. `preprocess_ABCD_get_metrics.ipynb` drops participants affected by issue 3 and adds a choice accuracy column as preprocessing. It then computes a) the SSD distributions for each participant, b) the accuracy and subsequent P(guess) for each SSD, and c) the inhibition function of the dataset. Each of these is saved to abcd_data/, but this is hidden from github.
  
2. `run_simulate.batch` runs `simulate.py`, which simulates data using each of the 3 models defined in Bissett, Hagen et al. (2020). The class which implements the simulations is found in `utils.py`. The outputs are saved to simulated_data/
  
3. `run_compute_SSRTS.batch` runs `compute_SSRTs.py`, which computes SSRT at each SSD 3 times: once while assuming context dependence, and once for each alternative model. It computes SSRT at each SSD on each of the 3 simulated datasets. These results are saved to ssrt_metrics/

4. `generate_results.ipynb` creates the figures relating to the simulations in the manuscipt and the expected SSRTs of individuals. In order, it generates a) the graph of SSRT by SSD by underlying model, b) a supplemental version of this graph which includes SSRT estimates in which the correct underlying distribution is assumed, c) the inhibition function of the simulate datasets as well as the ABCD dataset, d) rank correlations of expected individual SSRTS across the simulated datasets, and e) a plot of the proportional strength of the go drift rate as a function of SSD, used for the _slowed drift rate_ model. Figures are saved to figures/