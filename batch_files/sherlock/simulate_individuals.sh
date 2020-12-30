for mu_suffix in SSRTscale-85
do
    for iter in {0..4}
    do
        eval "sbatch ${mu_suffix}/sherlock_run_sims_iter${iter}.batch"
    done
done