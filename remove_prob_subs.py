from glob import glob
import os

# subs with P(respond|signal) == 1 | 0
problem_subs = ['0E4CT74P', '0LA6MBBY', '1PN1MPWF', '2L2B6VXP', '3FJ6DHVU',
                '4DVGGJE9', '4HWVY2FD', '5DJB8FFX', '7YA52BU1', '7ZRUXUX9',
                '8VA4L6RD', 'AB3XR5P1', 'AF517NF3', 'B3438R23', 'C2ZW147D',
                'E9KNUZDH', 'EC1NEE93', 'F86XUHXD', 'K9MXKUUB', 'KDZR86PA',
                'KP8W0VHU', 'L3258K1X', 'L5W5CDJU', 'L9NJJJBZ', 'LW0J7PJB',
                'MR60CAWU', 'T5EXYEGX', 'TY02C986', 'U1THG28C', 'YK2J75TH',
                'vcahyykd']

print('removing sim files')
for sub in problem_subs:
    files = glob(f'simulated_data/individual_data_*/*/*{sub}*.csv')
    print(sub, len(files))
    for f in files:
        os.remove(f)

print('removing ssrt files')
for sub in problem_subs:
    files = glob(f'ssrt_metrics/individual_metrics*/*{sub}*.csv')
    print(sub, len(files))
    for f in files:
        os.remove(f)
