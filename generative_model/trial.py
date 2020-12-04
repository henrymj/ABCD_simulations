# class for a trial

import numpy as np
from scipy.stats import exponnorm
from accumulator import Accumulator


def replace_none(x, replacement=np.inf):
    return(x if x is not None else replacement)


# hand-tuned grading function for mu_go
def log_mu_go(mu_go, SSD, max_SSD=550):
    SSD = min(SSD, max_SSD)
    return(0 if SSD == 0 else max(0, np.log(SSD / max_SSD) / 4 + 1 * mu_go))


def init_default_params():
    """return default parameters for a trial
    """
    return({'mu': {'go': 0.3, 'stop': 0.5},
            'max_time': 1000,
            'mu_delta_incorrect': 0.2,
            'nondecision': {'go': 50, 'stop': 50},
            'noise_sd': {'go': 2.2, 'stop': 2.2},
            'threshold': 100,
            'ntrials': {'go': 10000, 'stop': 2000},
            'mu_go_grader': None,
            'p_guess': None,
            'exgauss_params': None
            })


def fix_params(params):
    """if any parameter keys are missing, add them with None as value

    Args:
        params (dict): parameter dictionary
    """
    default_params = init_default_params()
    for k in default_params:
        if k not in params:
            params[k] = None
    return(params)


def sample_guess_rt(exgauss_params):
    return(
        exponnorm.rvs(exgauss_params['K'],
                      exgauss_params['loc'],
                      exgauss_params['scale']))


class Trial:
    def __init__(self, SSD=None, params=None, verbose=False, **kwargs):
        self.SSD = SSD
        if params is None:
            # use dummy settings for testing
            # NOTE: mu_delta_incorrect is a multiplier on mu_go
            self.params = init_default_params()
        else:
            self.params = params
        for key, value in kwargs.items():
            self.params[key] = value
        self.params = fix_params(self.params)

        if verbose:
            print(self.params)

        self.trial_type = 'go' if self.SSD is None else 'stop'
        self.rt_ = None

    def simulate(self, trial_params=None, verbose=False, **kwargs):
        if trial_params is None:
            trial_params = self.params
        for key, value in kwargs.items():
            trial_params[key] = value

        if verbose:
            print(trial_params)

        # first see if it's a fast guess, and if so then sample an RT and accuracy
        if trial_params['p_guess'] is not None:
            if trial_params['exgauss_params'] is None:
                raise Exception('Exgauss params must be specified if using the guessing model')

            if self.trial_type == 'go':
                p_fast_guess = trial_params['p_guess']['go']
            else:
                p_fast_guess = trial_params['p_guess']['stop'][self.SSD]
            if np.random.rand() < p_fast_guess:
                self.rt_ = sample_guess_rt(trial_params['exgauss_params'])

            self.correct_ = np.random.rand() < 0.5  # assume unbiased guessing
            return(self.rt_, self.correct_)

        # apply mu_go grader on stop trials
        if self.trial_type == 'stop' and trial_params['mu_go_grader'] == 'log':
            mu_go = log_mu_go(trial_params['mu']['go'], self.SSD)
        else:
            mu_go = trial_params['mu']['go']

        accumulator = Accumulator(
            mu=mu_go,
            noise_sd=trial_params['noise_sd']['go'],
            starting_point=trial_params['nondecision']['go'],
            max_time=trial_params['max_time'])
        correct_rt = accumulator.threshold_accumulator(threshold=trial_params['threshold'])

        accumulator = Accumulator(
            mu=trial_params['mu']['go'] * trial_params['mu_delta_incorrect'],
            noise_sd=trial_params['noise_sd']['go'],
            starting_point=trial_params['nondecision']['go'],
            max_time=trial_params['max_time'])
        incorrect_rt = accumulator.threshold_accumulator(threshold=trial_params['threshold'])

        # use replace_none to deal with cases where accumulator doesn't reach threshold
        if correct_rt is None and incorrect_rt is None:
            self.rt_ = None
            self.correct_ = None
        else:
            self.rt_ = np.min([replace_none(correct_rt),
                               replace_none(incorrect_rt)])

            self.correct_ = replace_none(correct_rt) < replace_none(incorrect_rt)

        if self.trial_type == 'stop':
            accumulator = Accumulator(
                mu=trial_params['mu']['stop'],
                noise_sd=trial_params['noise_sd']['stop'],
                starting_point=self.SSD + trial_params['nondecision']['stop'],
                max_time=trial_params['max_time'])
            stop_rt = accumulator.threshold_accumulator(threshold=trial_params['threshold'])
            # if stop process wins, set RT to None to reflect successful stop
            if self.rt_ is not None and stop_rt is not None and stop_rt < self.rt_:
                self.rt_ = None
                self.correct_ = None
            if verbose:
                print()

        return(self.rt_, self.correct_)


if __name__ == '__main__':
    trial = Trial(mu={'go': 0, 'stop': 0}, verbose=True)

    trial.simulate(verbose=True)
