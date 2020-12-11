import numpy as np
import pandas as pd


class SimulateData():

    def __init__(self, model='independent_race',
                 variable_mu_stop=False,
                 trigger_failures=False,
                 guesses=False,
                 grade_mu_go=False,
                 grade_mu_stop=False
                 ):
        self.model = model
        self.variable_mu_stop = variable_mu_stop
        self.trigger_failures = trigger_failures
        self.guesses = guesses
        self.grade_mu_go = grade_mu_go
        self.grade_mu_stop = grade_mu_stop

        trial_iterators = {
            'independent_race': self._independent_race_trial,
            'interactive_race': self._interactive_race_trial,
            'blocked_input': self._blocked_input_trial
        }
        self._trial_iter = trial_iterators[model]

    def simulate(self, params=None, method='fixed'):
        assert method in ['fixed', 'tracking']
        params = params if params else {}
        params = self._init_params(params)
        data_dict = self._init_data_dict()
        self._set_n_trials(params)
        self._set_n_guesses(params)
        if method == 'fixed':
            for ssd_idx, SSD in enumerate(params['SSDs']):
                data_dict = self._simulate_guesses(data_dict, params, SSD)
                data_dict = self._simulate_stop_trials(data_dict, params,
                                                       SSD)
        if method == 'tracking':
            data_dict = self._simulate_tracking_stop_trials(data_dict, params)
        data_dict = self._simulate_go_trials(data_dict, params)

        return self._convert_data_to_df(data_dict)

    def _accumulate(self, trial, mu, start_time, noise_sd):
        start_time = int(start_time)
        accum = np.zeros(trial['max_time'])
        accum_period = trial['max_time'] - start_time
        drift = np.ones(accum_period) * mu
        noise = np.random.randn(accum_period) * noise_sd
        accum[start_time:trial['max_time']] = np.cumsum(drift + noise)

        # retroactively include floor
        negative_spots = np.where(accum < 0)[0]
        for neg_idx in negative_spots:
            # the following is true for the first index,
            # but not necessarily afterwards
            if accum[neg_idx] < 0:
                accum[neg_idx:] += -(accum[neg_idx])

        # threshold for RT
        exceed_threshold = np.where(accum > trial['threshold'])[0]
        if len(exceed_threshold) > 0:
            rt = np.min(exceed_threshold)
        else:
            rt = np.nan
        return rt

    def _simulate_tracking_stop_trials(self, data_dict, params):
        SSD = params['tracking_start_ssd']
        for trial_idx in range(params['n_trials_tracking_stop']):
            stop_init_time = SSD + params['nondecision_stop']
            trial = self._init_trial_dict(params, trial_idx,
                                          SSD=SSD,
                                          stop_init_time=stop_init_time)
            # if it's a guess trial, guess, otherwise accumulate
            if np.random.rand() < self._p_guess_stop_dict[SSD]:
                # sample and grab the single rt
                go_rt = params['guess_function'](1)[0]
            else:
                go_rt = self._accumulate(trial,
                                         trial['mu_go'],
                                         trial['nondecision_go'],
                                         trial['noise_go']
                                         )
            stop_rt = self._accumulate(trial,
                                       trial['mu_stop'],
                                       trial['stop_init_time'],
                                       trial['noise_stop']
                                       )

            out_rt = np.nan
            if ~np.isnan(go_rt):
                out_rt = go_rt
            if ~np.isnan(go_rt) and ~np.isnan(stop_rt) and stop_rt < go_rt:
                out_rt = np.nan
            trial['RT'] = out_rt
            data_dict = self._update_data_dict(data_dict, trial)

            # UPDATE SSD
            SSD = self._trackSSD(SSD, params, np.isnan(out_rt))

        return data_dict

    def _trackSSD(self, SSD, params, stop_success_bool):

        if stop_success_bool:
            SSD += params['tracking_ssd_step']
        else:
            SSD -= params['tracking_ssd_step']

        if SSD < params['tracking_min_ssd']:
            SSD = params['tracking_min_ssd']
        elif SSD > params['tracking_max_ssd']:
            SSD = params['tracking_max_ssd']
        else:
            SSD = SSD
        return SSD


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
                stop_rt = self._accumulate(trial,
                                           trial['mu_stop'],
                                           trial['stop_init_time'],
                                           trial['noise_stop']
                                           )
                out_rt = guess_RT
                # sampled guess could be too long, or
                # stop process may have beat it
                if (guess_RT > trial['max_time']) | (~np.isnan(stop_rt) and
                                                     (stop_rt < guess_RT)):
                    out_rt = np.nan
                trial['RT'] = out_rt
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
            go_rt = self._accumulate(trial,
                                     trial['mu_go'],
                                     trial['nondecision_go'],
                                     trial['noise_go']
                                     )
            if ~np.isnan(go_rt):
                trial['RT'] = go_rt

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

        go_rt = self._accumulate(trial,
                                 trial['mu_go'],
                                 trial['nondecision_go'],
                                 trial['noise_go']
                                 )

        stop_rt = self._accumulate(trial,
                                   trial['mu_stop'],
                                   trial['stop_init_time'],
                                   trial['noise_stop']
                                   )
        # assign RT
        out_rt = np.nan
        if ~np.isnan(go_rt):
            out_rt = go_rt
        if ~np.isnan(go_rt) and ~np.isnan(stop_rt) and stop_rt < go_rt:
            out_rt = np.nan
        trial['RT'] = out_rt

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
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] -
                    trial['inhibition_interaction']*stop_accum +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break

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
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] -
                    trial['inhibition_interaction']*stop_accum +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break

        return self._update_data_dict(data_dict, trial)

    def _convert_data_to_df(self, data_dict):
        data_df = pd.DataFrame.from_dict(data_dict)
        data_df['block'] = 0
        for rt_type in ['go', 'stop']:
            data_df['{}RT'.format(rt_type)] = np.where(
                data_df['condition'] == rt_type,
                data_df['RT'],
                np.nan)

        # checks to make sure splitting was correct
        go_rt_idx = data_df[data_df.goRT.notnull()].index
        stop_rt_idx = data_df[data_df.stopRT.notnull()].index
        all_rt_idx = data_df[data_df.RT.notnull()].index

        assert np.allclose(go_rt_idx.difference(stop_rt_idx), go_rt_idx)
        assert np.allclose(stop_rt_idx.difference(go_rt_idx), stop_rt_idx)
        assert np.allclose(stop_rt_idx.union(go_rt_idx), all_rt_idx)

        del data_df['RT']

        return data_df

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
        # set up probability dictionary for tracking model
        self._p_guess_stop_dict = {SSD: p for SSD, p in zip(params['SSDs'],
                                                       p_guess_per_SSD)}

    def _get_mu_stop(self, params, SSD):
        mu_stop = params['mu_stop']
        if self.grade_mu_stop and SSD is not None:
            mu_stop = self._log_grade_mu(mu_stop, SSD)
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
        if self.grade_mu_go and SSD is not None:
            mu_go = self._log_grade_mu(mu_go, SSD)
        return mu_go

    def _log_grade_mu(self, mu_go, SSD, max_SSD=550):
        if SSD > max_SSD:
            SSD = max_SSD
        return self._at_least_0((np.log(SSD/max_SSD)/4+1) * mu_go)

    # def _linear_mu_go(self, mu_go, SSD, max_SSD=550):
    #     if SSD > max_SSD:
    #         SSD = max_SSD
    #     return self._at_least_0((SSD/max_SSD) * mu_go)

    def _init_params(self, params=None):
        params = params.copy() if params else {}
        # TODO: move default dict to json, read in
        default_dict = {'mu_go': .2,
                        'mu_stop': .4,
                        'noise_go':  2,  # 1.13,
                        'noise_stop': 2,  # 1.75,
                        'threshold': 100,
                        'nondecision_go': 50,
                        'nondecision_stop': 50,
                        'inhibition_interaction': .5,
                        'SSDs': np.arange(0, 600, 50),
                        'n_trials_go': 1000,
                        'n_trials_stop': 1000,
                        'n_trials_tracking_stop': 10000,
                        'max_time': 3000,
                        'p_trigger_fail': 0,
                        'p_guess_go': 0,
                        'p_guess_stop': 0,
                        'guess_function': lambda x: np.random.uniform(
                            200, 400, x),
                        'tracking_start_ssd': 50,
                        'tracking_min_ssd': 0,
                        'tracking_max_ssd': 500,
                        'tracking_ssd_step': 50,
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
            }

    def _init_trial_dict(self, params, trial_idx,
                         SSD=None, stop_init_time=np.nan, condition='stop'):
        trial = {
                'condition': condition,
                'SSD': SSD,
                'trial_idx': trial_idx,
                'mu_go': self._get_mu_go(params, SSD),
                'mu_stop': self._get_mu_stop(params, SSD),
                'stop_init_time': stop_init_time,
                'noise_go': params['noise_go'],
                'noise_stop': params['noise_stop'],
                'nondecision_go': params['nondecision_go'],
                'inhibition_interaction': params['inhibition_interaction'],
                'threshold': params['threshold'],
                'max_time': params['max_time'],
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
