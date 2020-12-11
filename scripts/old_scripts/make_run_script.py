# load subject codes

subcodes = ...

# create subject sets of particular length

nsubs_per_set = ...

subject_sets = [['sub1', 'sub2'], ...]

# create run script
with open('run_all_sims.sh', 'w') as f:
    for subject_set in subject_sets:
        sublist = ' '.join(subject_set)	
        f.write(f'python simulate_individual.py --subjects {sublist}\n'
