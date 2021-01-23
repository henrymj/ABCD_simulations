# Investigating violations of the assumption of context independence in the ABCD dataset.

## Overview (From the Supplement):

The goal of these simulations was to investigate the effect that different generating mechanisms could have on individual difference in the ABCD dataset, and stop signal data at large. The following is an overview of the process. 

We proposed three frameworks that may explain the context dependence in the ABCD data. Theindependent race model (Logan & Cowan, 1984; Logan et al., 2014) was used as a baseline model for comparison, and modified to produce .three alternative models which instantiated context dependence: 1) slowed go processing: the go drift rate is reduced by shorter go stimulus presentations at short SSDs, 2) guessing: a propensity to guess at shorter go stimulus presentations at short SSDs, and 3) confusion: both the go and stop drift rates are modulated by shorter go stimulus presentations at short SSDs. A maximum response time threshold of 3000ms was used to match the response window of the ABCD task.

We modeled slowed drift rates at shorter SSDs with the following formula at SSDs 0-550ms:  
```𝜇 @SSD = max(log(SSD/550)/4 + 1, 0) * 𝜇 ```

Where 𝜇 is the drift rate. This formula results in a 0 drift rate for SSDs of 0ms (as there is no exogenous go stimulus), and requires at least 10ms of go stimulus presentation to reach a positive value (though SSDs in ABCD are all a factor of 50ms). For SSDs greater than 10ms, the go drift rate rapidly approaches a normal rate with longer go stimulus presentations. This function (see Figure S1) was applied to modulate the go drift rate in the slowed go processing model and both go and stop drift rates in the confusion model.

To model guessing, a multistep process was applied. First, recognizing that stop-failure RTs for trials with an SSD of 0ms must be the result of guesses given that participants did not receive any go stimulus presentation on which to base their decisions (and evidenced by the fact that choice accuracy at that SSD was at chance levels), an exGaussian distribution was fit to those RTs to generate a guessing distribution, and a sampling function was built. We resampled whenever a sample was negative.  

Following this, a probability of guessing for trials at each SSD was computed. To do so, the following simplifying assumptions were made: stop-failure RTs at SSDs greater than 0ms are the result of mixing the go RT distribution with the guessing distribution outlined in the preceding paragraph, no-stop-signal trials do not include guesses, and the choice accuracy of stop-failure trials for a given SSD is the result of this mixing. Therefore, the probability of guessing at a given SSD was found by solving the following formula:


```ACC_ssd = P(guess|ssd) * ACC@SSD==0 + (1 - P(guess|ssd)) * ACCgo```

At each SSD, P proportion of simulated trials were sampled from the guess distribution, and 1-P proportion of simulated trials were taken from the no-stop-signal RT distribution. This simulation paradigm was created to investigate the effects of the ABCD stop task design issues on both mean SSRT estimates and individual differences. To provide all simulation details at once, the investigation of individual differences is described first.

To create hypothetical subjects and investigate the effects of the design issues on individual differences, the mean go RT and SSRT were computed for every ABCD subject, creating distributions that could be fit and sampled from. In addition, SSD distributions were collected from subjects by subsetting to stop trials with an SSD between 0 and 500, inclusive, and getting the proportion of stop trials within the subset at each SSD. These metrics were not collected for subjects if they experienced issue 3, or if their probability of responding given a stop signal (P(respond|signal)) was equal to 0 or 1, which indicates degenerate performance, leaving 8,207 subjects. Normal distributions were fit to the go RT and SSRT distributions in order to retrieve the mean and variance. However, because the SSRT values were contaminated by the issues described above, a range of hypothetical variances for the SSRT were tested in order to get a sense for how the underlying SSRT distribution interacted with the design issues described above to affect individual differences.

Across 4 conditions of between-subject SSRT variance (85ms, 25ms, 5ms, and 0ms), the following simulations occured. As discussed in the main text, we do not have a trustworthy estimate of individual subject SSRT, and therefore we do not have a trustworthy estimate of between-subject variability in SSRT. We chose these values to capture a range that varied from values that largely preserved SSRT as an individual difference metric (SD = 85ms) down to the limit of no variability such that any preservation of individual differences is driven by differences in go RT and SSD. We simulated 8,207 theoretical individuals. For each theoretical individual, a theoretical go RT and SSRT were independently sampled from the distributions described above, with resampling until both values were greater than 60ms (i.e. at least 10ms greater than the models’ nondecision time of 50ms). These were paired with a unique subject’s SSD distribution, and converted into drift rates (𝜇) using the following formula:

```𝜇 = threshold/({ssrt or rt} - nondecision time)```

Where the threshold was set to 100 and the nondecision time to 50ms.

Each theoretical subject’s drift rates were then inserted as parameters to the 4 competing models, and 2 sets of simulations occurred. First, a fixed SSD approach was applied, with 2,500 trials being simulated for each SSD in the range of 0ms to 500ms, with 50ms steps, inclusive. Second, a traditional 1 up 1 down tracking algorithm (Levitt, 1971) was applied, with 25,000 stop trials being simulated. In the staircase approach, the initial SSD was 50ms to match the initial SSD in the ABCD dataset, but the maximum SSD was 500ms, which is different from the max of 900ms in the ABCD dataset. In both cases, 5,000 go trials were simulated. 

Using the fixed SSD simulations, SSRT was computed at each SSD between 0 and 500 for each subject and each generating model, using the integration with replacement method, following the recommendations of Verbruggen et al. (2019). In addition, SSRT was recomputed at each SSD while using each alternative model to generate different underlying distributions for the go process on stop trials (this must be done at the SSD level because the alternative models predict different underlying distributions at different SSDs). For the two models with slowed drift rates, 5,000 go RTs were generated for each SSD. For the guessing model, the original go RTs were augmented with a number of sampled guess RTs such that the proportion of guesses to non-guesses matched that found in the second formula. For example, if the proportion of guesses at a given SSD was found to be 75%, then 15,000 guess RTs would be sampled to combine with the original 5,000 go RTs, creating an RT distribution of which 75% were guesses. These methods allowed us to compare SSRT estimates across different generating models while always making the assumption of context independence. 

First, SSRT was computed on the whole of each simulated individual’s fixed-SSD data. Second, we sampled an SSD distribution from a real ABCD subject, and we estimated a simulated mean SSRT by calculating a weighted sum of SSRTs across SSDs: 

```SSRT_indiv = SUM(P(ssd|indiv) * SSRT_(ssd&indiv))```

Third, an SSRT was computed on the individual’s tracking based simulation data. Rank correlations between alternative models and the standard independent race model using the same SSRT method were compared to describe the minimum and mean rank correlations within the text.

To investigate the effects of the ABCD stop task’s design issues on mean SSRT estimates (e.g., Figure 5), the above simulations were repeated while setting both the SSRT and go RT between-subject variance to 0, assigning the mean of both values to the 8,207 simulated subjects. Because each of the 8,207 simulated subjects had the same parameters, this is equivalent to having a single simulated dataset with a large number of trials. For each subject, 3 SSRT estimates were again computed (weighted, fixed, tracking) per underlying model; these were then averaged across subjects. The means from the tracking-based SSRTs are reported in the manuscript, but the fixed-SSD and weighted SSRTs are qualitatively identical, and are presented in the results notebooks.

## Viewing Results / Numbers Referenced in MS:
The notebooks at the top level compute or display all numbers referenced in the manuscript.
- `0preprocess_ABCD_get_metrics.ipynb` shows the number of subjects kept, the RT and SSRT distributions, and other metrics.
- `1display_results_SSRTscale-*.ipynb` shows the results of the simulations for the given SSRT variance. This includes scatter plots and correlations of the different SSRTs
- `2results_across_SSRTscales.ipynb` shows the correlations between SSRTs for generating table 1 and comparing the interaction between the ABCD stop task's design issues and the underlying variance in SSRT variance.

## Running Pipeline:
*Note: the SSRT analyses rely on `stopsignalmetrics` (https://doi.org/10.5281/zenodo.4458764), a Poldrack lab python package for standard analyses of stop signal datasets, including SSRT calculation.

0. The most important files are located in `scripts/`: `simulate_individuals.py`, `compute_indiv_SSRTs.py`, `get_individual_results.py`, and `utils.py`.

1. Setup:  
`0preprocess_ABCD_get_metrics.ipynb` drops participants affected by issue 3 and adds a choice accuracy column as preprocessing. It then computes a) SSRT per participant in order to get mu_go and mu_stop from their mean RT and SSRT, b) the SSD distributions for each participant, c) the accuracy and subsequent P(guess) for each SSD, and d) the inhibition function of the dataset. Each of these is saved to abcd_data/, but this is hidden from github.
It edits `batch_files/sherlock/SSRTscale-*/sherlock_run_sims_iter*.batch` for running jobs in parallel on Sherlock. The code was originally developed for deployment on TACC, so there is some legacy slurm files in `batch_files/TACC`.
  
2. Simulating:   
In `batch_files/sherlock`, run `bash simulate_individuals.sh`. This will run `simulate_individuals.py` through the intermediary batch files that are edited above. 
- After the simulations finish, in `batch_files/`, run `python generate_remaining_sim_cmds.py` to look for subs that did not complete the simulations (due to time or resource limits) and edit the batch files again, though this should not occur and is just for convenience.

3. Computing SSRTs:  
In `batch_files/`, run `python generate_SSRT_cmds.py` to find subjects that have been simulated but don't have ssrt files. It edits `batch_files/sherlock/SSRTscale-*/sherlock_run_ssrt_iter*.batch` for running jobs on Sherlock. In `batch_files/sherlock`, run `bash compute_ssrts.sh` which will read run `compute_indiv_SSRTs.py` through the intermediary batch files above, computing SSRTs and getting related metrics per SSD per simulation type x SSRT assumption. 
- Rerunning `generate_SSRT_cmds.py` will update the batch files to get SSRT for any subjects which didn't complete.

4. Generate Results:  
In `batch_files/sherlock`, run `sbatch get_results.batch` to run `get_individual_results.py`. This reads in the individual ssrt files generated in step 3 and creates the figures along with the expected SSRT for each simulation type x SSRT method per subject.

5. Review Results:  
Run `1display_results_SSRTscale-*.ipynb` and `2results_across_SSRTscales.ipynb` to view the figures and examine the rank correlations across the simulation type x SSRT method combination.