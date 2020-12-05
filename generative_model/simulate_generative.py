from os import times
import numpy as np
import pandas as pd


def at_least_0(num):
    return(np.max([0., num]))


# class run to run generative simulation
class SimulateGenerative():
    def __init__(self, model='independent_race',
                 variable_mu_stop=False,
                 trigger_failures=False,
                 guesses=False,
                 mu_go_grader=None):
        self.params = {
            'mu_go': 0.5,  # temp set from a randomly chosen subject
            'mu_stop': 0.5,
            'n_trials': 100,
            'max_time': 1000,
            'nondecision_go': 50,
            'noise_go': 3.2,
            'threshold': 100
        }

    def _simulate_accumulator(self, starting_point, mu, noise, threshold):
        if trial_params is None:
            trial_params = self.params

        go_accum = np.zeros(trial_params['max_time'])
        # for period after nondecision time, add noise to mu_go
        accumulation_period = trial_params['max_time'] - trial_params['nondecision_go']
        mu_accumulator = np.cumsum(np.ones(accumulation_period) * trial_params['mu_go'])
        accumulated_noise = np.cumsum(np.random.randn(accumulation_period) * trial_params['noise_go'])
        go_accum[trial_params['nondecision_go']:] = mu_accumulator + accumulated_noise
        exceed_threshold = np.where(go_accum > trial_params['threshold'])
        try:
            rt = np.min(exceed_threshold[0])
        except ValueError:
            rt = None

        return(rt, go_accum)


    def _simulate_trials(self):
        for trial_idx in range(self.n_trials):
            pass

if __name__ == '__main__':
    s = SimulateGenerative()
    rt, go_accum = s._simulate_go_accumulator(None)