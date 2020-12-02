## generate data for a subject

from collections import namedtuple
import numpy as np
import json

from simulate import generate_exgauss_sampler_from_fit
from utils import SimulateData


def get_exgauss_sampler():
    SSD0_RTs = np.random.randn(100) * 0.5
    return(generate_exgauss_sampler_from_fit(SSD0_RTs))


if __name__ == '__main__':
    Args = namedtuple('args', 'p_stop, n_trials, model_type, abcd_dir')
    args = Args(p_stop = 1/6,
                n_trials = 1000,
                model_type = 'standard',
                abcd_dir = 'abcd_data')

    with open('%s/individual_mus.json' % args.abcd_dir) as json_file:
        mus_dict = json.load(json_file)

    simulator_dict = {
        'standard': SimulateData(),
        'guesses': SimulateData(guesses=True),
        'graded_mu_go_log': SimulateData(mu_go_grader='log'),
    }

    params = {
        'n_trials': args.n_trials,
        'p_stop': args.p_stop,
        'n_trials_stop': np.round(args.n_trials * args.p_stop).astype(int),
        'guess_function': get_exgauss_sampler(),
        'SSDs': np.arange(0, .95, .05),
        
    }
    sub = 'ZVW3HKWN' # TEMP KLUDGE
    params['mu_go'] = mus_dict[sub]['go']
    params['mu_stop'] = mus_dict[sub]['stop']
    params['p_guess_stop'] = [0 for i in params['SSDs']]

    params['n_trials_go'] = args.n_trials - params['n_trials_stop']

    data = simulator_dict[args.model_type].simulate()
