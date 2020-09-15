for method in vanilla guesses log linear
do
    sed  -e "s/{method}/$method/g" simulate_w_SSD_dists.batch | sbatch
done