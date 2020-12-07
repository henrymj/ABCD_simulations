from os import times
import numpy as np
import pandas as pd

from accumulator import Accumulator

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

    def _simulate_accumulator(self, starting_point=50, mu=0.3, 
                              noise_sd=0.3, max_time=1000):
        """simulate an accumulator process

        Args:
            starting_point ([type]): starting point for accumulation (nondecision time for go, SSD for stop), in millseconds
            mu ([type]): drift rate
            noise_sd ([type]): noise standard deviation
            max_time ([type]): maximum time point (in milliseconds)
        """

        accum = np.zeros(max_time)
        # for period after start, add noise to mu_go
        accumulation_period = max_time - starting_point
        mu_accumulator = np.cumsum(np.ones(accumulation_period) * mu)
        accumulated_noise = np.cumsum(np.random.randn(accumulation_period) * noise_sd)
        accum[starting_point:] = mu_accumulator + accumulated_noise
        return(accum)

    def _get_rt_from_accumulator(self, accum, threshold=100):
        exceed_threshold_abs = np.where(np.abs(accum) > threshold)
        if len(exceed_threshold_abs[0]) > 0:
            rt = np.min(exceed_threshold_abs[0])
        else:
            rt = None
        correct = accum[rt] >= threshold
        return(rt, correct)

    def _simulate_trial(self, params, SSD=None):
        if SSD is None:  # go trial
            pass

    def _simulate_trials(self):
        for trial_idx in range(self.n_trials):
            pass

if __name__ == '__main__':
    s = SimulateGenerative()
    accum = s._simulate_go_accumulator(None)
    rt, correct = 