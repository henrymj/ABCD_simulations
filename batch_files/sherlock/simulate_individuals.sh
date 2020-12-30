for mu_suffix in SSRTscale-85 SSRTscale-25 SSRTscale-0 SSRTscale-0_RTscale-0
do
    for iter in {0..4}
    do
        eval "sbatch ${mu_suffix}/sherlock_run_sims_iter${iter}.batch"
    done
done