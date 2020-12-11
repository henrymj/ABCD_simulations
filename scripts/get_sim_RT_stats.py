import pandas as pd
from glob import glob

sim_files = glob('/oak/stanford/groups/russpold/users/henrymj/ABCD_simulations/simulated_data/individual_data/standard_*.csv')

sim_rt_dict = {}
for sim_file in sim_files:
    subid = sim_file.split('_')[-1].replace('.csv', '')
    sim_rt_dict[subid] = pd.read_csv(sim_file)['goRT'].describe().loc[['mean', 'std']]

pd.DataFrame(sim_rt_dict).T.add_prefix('sim_rt_').to_csv('/oak/stanford/groups/russpold/users/henrymj/ABCD_simulations/simulated_data/sim_rt_meanStd.csv')
