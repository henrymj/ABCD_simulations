# class for a trial

from accumulator import Accumulator


class Trial:
    def __init__(self, SSD=None, params=None, verbose=False, **kwargs):
        if params is None:
            # use dummy settings for testing
            self.params = {
                'mu': {'go': 0.5, 'stop': 0.5},
                'max_time': 1000,
                'nondecision': {'go': 50, 'stop': 50},
                'noise_sd': {'go': 3.2, 'stop': 3.2},
                'threshold': 100
            }
        else:
            self.params = params
        for key, value in kwargs.items():
            self.params[key] = value

        if verbose:
            print(self.params)

        self.SSD = SSD
        self.trial_type = 'go' if self.SSD is None else 'stop'
        self.rt_ = None

    def simulate(self, trial_params=None, verbose=False, **kwargs):
        if trial_params is None:
            trial_params = self.params
        for key, value in kwargs.items():
            trial_params[key] = value

        if verbose:
            print(trial_params)

        if self.trial_type == 'go':
            starting_point = trial_params['nondecision'][self.trial_type]
        elif self.trial_type == 'stop':
            starting_point = self.SSD
        else:
            raise Exception(f'unknown trial type: {self.trial_type}')

        accumulator = Accumulator(
            mu=trial_params['mu']['go'],
            noise_sd=trial_params['noise_sd']['go'],
            starting_point=starting_point,
            max_time=trial_params['max_time'])
        self.rt_ = accumulator.threshold_accumulator(threshold=trial_params['threshold'])

        if self.trial_type == 'stop':
            accumulator = Accumulator(
                mu=trial_params['mu']['stop'],
                noise_sd=trial_params['noise_sd']['stop'],
                starting_point=starting_point,
                max_time=trial_params['max_time'])
            stop_rt = accumulator.threshold_accumulator(threshold=trial_params['threshold'])
            # if stop process wins, set RT to None to reflect successful stop
            if stop_rt < self.rt_:
                self.rt_ = None

        return(self.rt_)


if __name__ == '__main__':
    trial = Trial(mu={'go': 0, 'stop': 0}, verbose=True)

    trial.simulate(verbose=True)
