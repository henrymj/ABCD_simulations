import numpy as np
import pandas as pd


class SimulateData():

    def __init__(self, model='independent_race',
                 variable_mu_stop=False,
                 trigger_failures=False,
                 guesses=False,
                 mu_go_grader=None):
        self.model = model
        self.variable_mu_stop = variable_mu_stop
        self.trigger_failures = trigger_failures
        self.guesses = guesses
        trial_iterators = {
            'independent_race': self._independent_race_trial,
            'interactive_race': self._interactive_race_trial,
            'blocked_input': self._blocked_input_trial
        }
        self._trial_iter = trial_iterators[model]

        self._mu_go_grader = None
        if mu_go_grader:
            mu_go_graders = {
                'log': self._log_mu_go,
                'linear': self._linear_mu_go,
            }
            assert mu_go_grader in ['log', 'linear']
            self._mu_go_grader = mu_go_graders[mu_go_grader]

    def simulate(self, params={}):
        params = self._init_params(params)
        data_dict = self._init_data_dict()
        self._set_n_trials(params)
        self._set_n_guesses(params)
        for ssd_idx, SSD in enumerate(params['SSDs']):
            data_dict = self._simulate_guesses(data_dict, params, SSD)
            data_dict = self._simulate_stop_trials(data_dict, params,
                                                   SSD)
        data_dict = self._simulate_go_trials(data_dict, params)

        # convert to dataframe
        data_df = pd.DataFrame.from_dict(data_dict)
        data_df['block'] = 0
        for rt_type in ['go', 'stop']:
            data_df['{}RT'.format(rt_type)] = np.where(
                data_df['condition'] == rt_type,
                data_df['RT'],
                np.nan)
        del data_df['RT']

        return data_df

    def _simulate_guesses(self, data_dict, params, SSD):
        if SSD is None:  # go trials
            n_guess = int(self._n_guess_go)
        else:
            n_guess = int(self._n_guess_stop[SSD])
        if n_guess == 0:
            return data_dict
        guess_RTs = params['guess_function'](
            n_guess
        )
        stop_init_time = SSD + params['nondecision_stop'] if\
            (SSD is not None) else np.nan
        for trial_idx, guess_RT in enumerate(guess_RTs):
            trial = self._init_trial_dict(params, trial_idx,
                                          SSD=SSD,
                                          stop_init_time=stop_init_time)
            if SSD is not None:
                stop_accum = 0
                for time in range(1, trial['max_time']+1):
                    if time >= trial['stop_init_time']:
                        stop_accum = self._at_least_0(
                            stop_accum + trial['mu_stop'] +
                            np.random.normal(loc=0, scale=trial['noise_stop'])
                        )
                        trial['process_stop'].append(stop_accum)
                    if stop_accum > trial['threshold']:
                        break

                if guess_RT <= time:
                    trial['RT'] = guess_RT
            else:
                if guess_RT <= trial['max_time']:
                    trial['RT'] = guess_RT
            data_dict = self._update_data_dict(data_dict, trial)
        return data_dict

    def _simulate_go_trials(self, data_dict, params):
        data_dict = self._simulate_guesses(data_dict, params, None)
        for trial_idx in range(int(self._n_guess_go),
                               self._n_trials_go):
            trial = self._init_trial_dict(params, trial_idx, condition='go')
            go_accum = 0
            stop_accum = 0
            for time in range(1, trial['max_time']+1):
                if time >= trial['nondecision_go']:
                    go_accum = self._at_least_0(
                        go_accum + trial['mu_go'] +
                        np.random.normal(loc=0, scale=trial['noise_go'])
                    )
                    trial['process_go'].append(go_accum)
                if go_accum > trial['threshold']:
                    trial['RT'] = time
                    break

            trial['accum_go'] = go_accum
            trial['accum_stop'] = stop_accum
            data_dict = self._update_data_dict(data_dict, trial)
        return data_dict

    def _simulate_stop_trials(self, data_dict, params, SSD):
        stop_init_time = SSD + params['nondecision_stop']
        for trial_idx in range(int(self._n_guess_stop[SSD]),
                               int(self._n_trials_stop[SSD])):
            trial = self._init_trial_dict(params, trial_idx,
                                          SSD=SSD,
                                          stop_init_time=stop_init_time)
            data_dict = self._trial_iter(data_dict, trial)
        return data_dict

    def _independent_race_trial(self, data_dict, trial):
        go_accum = 0
        stop_accum = 0
        for time in range(1, trial['max_time']+1):
            if time >= trial['stop_init_time']:
                stop_accum = self._at_least_0(
                    stop_accum + trial['mu_stop'] +
                    np.random.normal(loc=0, scale=trial['noise_stop'])
                )
                trial['process_stop'].append(stop_accum)
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
                trial['process_go'].append(go_accum)
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break
            if stop_accum > trial['threshold']:
                break

        trial['accum_go'] = go_accum
        trial['accum_stop'] = stop_accum
        return self._update_data_dict(data_dict, trial)

    def _interactive_race_trial(self, data_dict, trial):
        go_accum = 0
        stop_accum = 0
        for time in range(1, trial['max_time']+1):
            if time >= trial['stop_init_time']:
                stop_accum = self._at_least_0(
                    stop_accum + trial['mu_stop'] +
                    np.random.normal(loc=0, scale=trial['noise_stop'])
                )
                trial['process_stop'].append(stop_accum)
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] -
                    trial['inhibition_interaction']*stop_accum +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
                trial['process_go'].append(go_accum)
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break
        trial['accum_go'] = go_accum
        trial['accum_stop'] = stop_accum
        return self._update_data_dict(data_dict, trial)

    def _blocked_input_trial(self, data_dict, trial):
        go_accum = 0
        stop_accum = 0
        for time in range(1, trial['max_time']+1):
            if time >= trial['stop_init_time']:
                stop_accum = self._at_least_0(
                    trial['mu_stop'] + np.random.normal(
                        loc=0, scale=trial['noise_stop'])
                )
                trial['process_stop'].append(stop_accum)
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] -
                    trial['inhibition_interaction']*stop_accum +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
                trial['process_go'].append(go_accum)
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break

        trial['accum_go'] = go_accum
        trial['accum_stop'] = stop_accum
        return self._update_data_dict(data_dict, trial)

    def _set_n_trials(self, params):
        num_SSDs = len(params['SSDs'])
        n_trials_stop = params['n_trials_stop']
        if type(params['n_trials_stop']) in [float, int]:
            n_trials_stop = [params['n_trials_stop']] * num_SSDs
        elif type(params['n_trials_stop']) in [list, np.ndarray]:
            if len(params['n_trials_stop']) == 1:
                n_trials_stop = params['n_trials_stop'] * num_SSDs
            else:
                n_trials_stop = params['n_trials_stop']

        assert(len(n_trials_stop) == num_SSDs)

        self._n_trials_go = params['n_trials_go']
        self._n_trials_stop = {SSD: n for SSD, n in
                               zip(params['SSDs'], n_trials_stop)}

    def _set_n_guesses(self, params):
        # TODO: ADD ASSERTIONS TO CHECK FOR CORRECT USES, clean up!!!
        # TODO: allow for guessing on go trials
        num_SSDs = len(params['SSDs'])
        if self.guesses:
            p_guess_go = params['p_guess_go']
            if type(params['p_guess_stop']) in [float, int]:
                p_guess_per_SSD = [params['p_guess_stop']] * num_SSDs
            elif type(params['p_guess_stop']) in [list, np.ndarray]:
                if len(params['p_guess_stop']) == 1:
                    p_guess_per_SSD = params['p_guess_stop'] * num_SSDs
                else:
                    p_guess_per_SSD = params['p_guess_stop']
            else:
                print('did not expect type {}'.format(
                      type(params['p_guess_stop'])))
        else:
            p_guess_per_SSD = [0] * num_SSDs
            p_guess_go = 0
        assert(len(p_guess_per_SSD) == num_SSDs)

        # TODO: clean up these lines? -
        # if 0 is returned, it's viewed as an int,
        # not a float, so it needs to be converted
        self._n_guess_go = self._at_least_0(
            np.rint(float(
                    p_guess_go * self._n_trials_go)))
        self._n_guess_stop = {SSD: self._at_least_0(
                                np.rint(float(p * self._n_trials_stop[SSD])))
                              for SSD, p in zip(params['SSDs'],
                                                p_guess_per_SSD)}

    def _get_mu_stop(self, params):
        mu_stop = params['mu_stop']
        if self.variable_mu_stop:
            mu_stop = mu_stop+np.random.normal(
                loc=0, scale=params['noise_stop']*.7)
        if self.trigger_failures and np.random.uniform(0, 1) <\
                params['p_trigger_fail']:
            mu_stop = 0
        return self._at_least_0(mu_stop)

    def _get_mu_go(self, params, SSD):
        # TODO: make more dynamic, pass max_SSD
        mu_go = params['mu_go']
        if self._mu_go_grader and SSD is not None:
            mu_go = self._mu_go_grader(mu_go, SSD)
        return mu_go

    def _log_mu_go(self, mu_go, SSD, max_SSD=550):
        if SSD > max_SSD:
            SSD = max_SSD
        return self._at_least_0((np.log(SSD/max_SSD)/4+1) * mu_go)

    def _linear_mu_go(self, mu_go, SSD, max_SSD=550):
        if SSD > max_SSD:
            SSD = max_SSD
        return self._at_least_0((SSD/max_SSD) * mu_go)

    def _init_params(self, params):
        # TODO: move default dict to json, read in
        default_dict = {'mu_go': .2,
                        'mu_stop': .4,
                        'noise_go':  3.5,  # 2,  # 1.13,
                        'noise_stop': 3,  # 2,  # 1.75,
                        'threshold': 100,
                        'nondecision_go': 50,
                        'nondecision_stop': 50,
                        'inhibition_interaction': .5,
                        'SSDs': np.arange(0, 600, 50),
                        'n_trials_go': 1000,
                        'n_trials_stop': 1000,
                        'max_time': 3000,
                        'p_trigger_fail': 0,
                        'p_guess_go': 0,
                        'p_guess_stop': 0,
                        'guess_function': lambda x: np.random.uniform(
                            200, 400, x),
                        'mu_go_grader': 'log'
                        }

        for key in default_dict:
            if key not in params:
                params[key] = default_dict[key]

        return params

    def _init_data_dict(self):
        return {
            'condition': [],
            'SSD': [],
            'trial_idx': [],
            'RT': [],
            'mu_go': [],
            'mu_stop': [],
            'accum_go': [],
            'accum_stop': [],
            'process_go': [],
            'process_stop': [],
            }

    def _init_trial_dict(self, params, trial_idx,
                         SSD=None, stop_init_time=np.nan, condition='stop'):
        trial = {
                'condition': condition,
                'SSD': SSD,
                'trial_idx': trial_idx,
                'mu_go': self._get_mu_go(params, SSD),
                'mu_stop': self._get_mu_stop(params),
                'stop_init_time': stop_init_time,
                'noise_go': params['noise_go'],
                'noise_stop': params['noise_stop'],
                'nondecision_go': params['nondecision_go'],
                'inhibition_interaction': params['inhibition_interaction'],
                'threshold': params['threshold'],
                'max_time': params['max_time'],
                'accum_go': np.nan,
                'accum_stop': np.nan,
                'process_go': [],
                'process_stop': [],
                'RT': np.nan
            }
        return trial

    def _update_data_dict(self, data_dict, update_dict):
        for key in data_dict.keys():
            data_dict[key].append(update_dict[key])
        return data_dict

    def _at_least_0(self, num):
        if num < 0:
            num = 0
        return num
