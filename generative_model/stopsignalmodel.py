import numpy as np
import json
from ssd import fixedSSD, trackingSSD
from stoptaskstudy import StopTaskStudy
from generative_utils import cleanup_metrics


class StopSignalModel:
    # class variable: allowed models
    implemented_models = ['basic',
                          'simpleguessing',
                          'scaledguessing',
                          'gradedmugo',
                          'gradedmuboth',
                          'fullabcd',
                          'cacaphony']

    def __init__(self, model, paramfile=None, parameters=None,
                 ssdtype='fixed', **kwargs):
        """Class for stop signal model

        Args:
            paramfile (str, optional): json file with parameters. Defaults to None.
            parameters (dict, optional): dictionary with parameters. Defaults to None.
                NOTE: parameters takes precendence over paramfile
            ssdtype (str, optional): [description]. Defaults to 'fixed'.
            **kwargs: additional parameters can be specified by keyword arguments
                NOTE: kwargs take precedence over arguments in paramfile or parameters

        """

        assert parameters is not None or paramfile is not None, "You must specify either parameters or paramfile"

        self.model = model
        self.paramfile = paramfile
        if paramfile is not None:
            with open(paramfile) as f:
                self.parameters = json.load(f)
        else:
            self.parameters = {}

        # add parameters from input dict
        self.input_parameters = {} if parameters is None else parameters
        for k, value in self.input_parameters.items():
            # deal with second-layer settings
            if isinstance(value, dict):
                if k not in self.parameters:
                    self.parameters[k] = {}
                self.parameters[k].update(value)
            else:
                self.parameters[k] = value

        # add parameters from keyword args
        self.parameters.update(kwargs)

        self.ssdtype = ssdtype
        self.setup_SSD_generator()

    def setup_SSD_generator(self):
        self.setup_default_ssd_params()
        if self.ssdtype == 'fixed':
            self.SSDfunc = fixedSSD(
                np.arange(self.parameters['min_ssd'],
                          self.parameters['max_ssd'] + 1,  # add 1 to include max
                          self.parameters['ssd_step']))
        elif self.ssdtype == 'tracking':
            self.SSDfunc = trackingSSD()
        else:
            raise Exception(f'SSD type {self.ssdtype} not recognized')

    def setup_default_ssd_params(self):
        if 'min_ssd' not in self.parameters:
            self.parameters['min_ssd'] = 0
        if 'max_ssd' not in self.parameters:
            self.parameters['max_ssd'] = 500
        if 'ssd_step' not in self.parameters:
            self.parameters['ssd_step'] = 50

    # use model label to set up parameters
    def setup_model_params(self, modelparams):
        # install ABCSMC parameters into full parameter set
        self.parameters['mu']['go'] = modelparams['mu_go']
        self.parameters['mu']['stop'] = modelparams['mu_go'] + modelparams['mu_stop_delta']
        self.parameters['mu_delta_incorrect'] = modelparams['mu_delta_incorrect']
        self.parameters['noise_sd'] = {'go': modelparams['noise_sd'],
                                       'stop': modelparams['noise_sd']}
        self.parameters['nondecision'] = {'go': int(modelparams['nondecision']),
                                          'stop': int(modelparams['nondecision'])}
        if 'pguess' in modelparams:
            self.parameters['p_guess']['go'] = modelparams['pguess']

    # takes in a dict of model parameters
    # returns a dict of peformance statistics
    def fit_transform(self, modelparams, verbose=False):
        """model

        Args:
            modelparams (dict): parameters passed it from ABCSMC

        Returns:
            results (dict): results from model
        """

        self.setup_model_params(modelparams)

        # install the parameters from the simulation
        # TBD
        #    if args.p_guess_file is not None:
        #        p_guess = pd.read_csv(args.p_guess_file, index_col=0)
        #        assert 'SSD' in p_guess.columns and 'p_guess' in p_guess.columns

        study = StopTaskStudy(self.SSDfunc, None, params=self.parameters)

        trialdata = study.run()
        trialdata['correct'] = trialdata.correct.astype(float)
        metrics = study.get_stopsignal_metrics()
        # summarize data - go trials are labeled with SSD of -inf so that
        # they get included in the summary
        presp_by_ssd = trialdata.groupby('SSD').mean().query('SSD >= 0').resp.values
        results = {}

        metrics = cleanup_metrics(metrics)
        if verbose:
            print(metrics)
        for k in ['mean_go_RT', 'mean_stopfail_RT', 'go_acc', 'sd_go_RT', 'sd_stopfail_RT']:
            results.update({k: metrics[k]})
        # need to separate presp values since distance fn can't take a vector
        for i, value in enumerate(presp_by_ssd):
            # occasionally there will be no trials for a particular SSD which gives NaN
            # we replace that with zero
            results[f'presp_{i}'] = 0 if np.isnan(value) else value

        for i, SSD in enumerate(trialdata.query('SSD >= 0').SSD.sort_values().unique()):
            accdata_for_ssd = trialdata.query(f'SSD == {SSD}').dropna()
            value = accdata_for_ssd.correct.dropna().mean()
            results[f'accuracy_{i}'] = 0 if np.isnan(value) else value
        self.results_ = results
        return(results)


if __name__ == '__main__':
    ssm = StopSignalModel('basic', paramfile='params/params_basic.json')
    assert ssm is not None
    modelparams = {
        'mu_delta_incorrect': 0.1083986991950653,
        'mu_go': 0.16778296718781244,
        'mu_stop_delta': 0.012833735851097903,
        'noise_sd': 2.511930980540796,
        'nondecision': 73.12196295189378}
    result = ssm.fit_transform(modelparams)
    assert result is not None
    assert all(item in result for item in [
        'mean_go_RT', 'mean_stopfail_RT', 'go_acc', 'sd_go_RT', 'sd_stopfail_RT'])
