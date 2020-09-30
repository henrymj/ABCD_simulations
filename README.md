# Investigating violations of the assumption of context independence in the ABCD dataset.

## Running:

1. `preprocess_ABCD_get_metrics.ipynb` drops participants affected by issue 3 and adds a choice accuracy column as preprocessing. It then computes a) the SSD distributions for each participant, b) the accuracy and subsequent P(guess) for each SSD, and c) the inhibition function of the dataset. Each of these is saved to abcd_data/
  
2. `run_simulate.batch` runs `simulate.py`, which simulates data using each of the 3 models defined in Bissett, Hagen et al. (2020). The class which implements the simulations is found in `utils.py`. The outputs are saved to simulated_data/
  
3. `run_compute_SSRTS.batch` runs `compute_SSRTs.py`, which computes SSRT at each SSD 3 times: once while assuming context dependence, and once for each alternative model. It computes SSRT at each SSD on each of the 3 simulated datasets. These results are saved to ssrt_metrics/

4. `generate_results.ipynb` creates the figures relating to the simulations in the manuscipt and the expected SSRTs of individuals. In order, it generates a) the graph of SSRT by SSD by underlying model, b) a supplemental version of this graph which includes SSRT estimates in which the correct underlying distribution is assumed, c) the inhibition function of the simulate datasets as well as the ABCD dataset, d) rank correlations of expected individual SSRTS across the simulated datasets, and e) a plot of the proportional strength of the go drift rate as a function of SSD, used for the _slowed drift rate_ model. Figures are saved to figures/