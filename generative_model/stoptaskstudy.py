# class for a single run of a stop study (i.e. a single subject)

import numpy as np
from trial import Trial
from collections import namedtuple
import pandas as pd
from stopsignalmetrics import StopData, SSRTmodel, PostStopSlow, Violations, StopSummary


# SSD generators
class fixedSSD:
    def __init__(self, SSDs, **kwargs):
        """[summary]

        Args:
            SSDs (list): a list of SSDs (in which case they are presented with equal frequency)
                TBD(?): or a dictionary containing the SSDs and their relative frequency
        """
        self.SSDs = np.array(SSDs)

    def generate(self, n_stop_trials):
        """generator for fixed SSD design
        - if number of SSDs is not a multiple of n_stop_trials, then extra SSDs are randomly selected

        Args:
            n_stop_trials (int): number of stop trials

        Yields:
            [type]: [description]
        """
        num_repeats = np.ceil(n_stop_trials / len(self.SSDs))
        SSDlist = np.repeat(self.SSDs, num_repeats)
        np.random.shuffle(SSDlist)
        SSDlist = SSDlist[:n_stop_trials]
        for i in range(n_stop_trials):
            yield SSDlist[i]


# main study class
class StopTaskStudy:
    def __init__(self, SSDgenerator, params=None, **kwargs):
        if params is None:
            self.params = {
                'mu': {'go': 0.3, 'stop': 0.5},
                'max_time': 1000,
                'mu_delta_incorrect': 0.2,
                'nondecision': {'go': 50, 'stop': 50},
                'noise_sd': {'go': 2.2, 'stop': 2.2},
                'threshold': 100,
                'ntrials': {'go': 10000, 'stop': 2000},
            }
        else:
            self.params = params
        for key, value in kwargs.items():
            self.params[key] = value

        self.SSDgenerator = SSDgenerator
        self.trialdata_ = None
        # create random subject code
        if 'subcode' not in self.params:
            self.params['subcode'] = np.random.randint(10e12)

    def run(self):

        Trialdata = namedtuple('Trialdata', 'trialtype, SSD, rt, resp, correct')
        trialdata = []

        # generate stop trials
        for i, SSD in enumerate(self.SSDgenerator.generate(self.params['ntrials']['stop'])):
            trial = Trial(SSD, self.params)
            rt, correct = trial.simulate()
            trialdata.append(Trialdata(
                trialtype='stop',
                SSD=SSD,
                rt=rt,
                resp=(rt is not None),
                correct=correct))

        # generate go trials
        SSD = -np.inf
        for _ in range(self.params['ntrials']['go']):
            # first simulate correct response racer
            trial = Trial(None, self.params)
            rt, correct = trial.simulate()

            trialdata.append(Trialdata(
                trialtype='go',
                SSD=SSD,
                rt=rt,
                resp=(rt is not None),
                correct=correct))
        self.trialdata_ = pd.DataFrame(data=trialdata)
        return(self.trialdata_)

    def get_stopsignal_metrics(self):
        trialdata = self.trialdata_
        # clean up data for stopsignal metrics analysis
        trialdata['subcode'] = study.params['subcode']
        trialdata['resp'] = trialdata.resp.astype(int)
        trialdata['correct_response'] = 1
        trialdata['block'] = 1
        trialdata['condition'] = trialdata['trialtype']
        trialdata['goRT'] = trialdata['rt']
        trialdata.loc[trialdata.trialtype == 'stop', 'goRT'] = None
        trialdata['stopRT'] = trialdata['rt']
        trialdata.loc[trialdata.trialtype == 'go', 'stopRT'] = None

        # compute metrics
        ssrt_model = SSRTmodel(model='all')
        self.metrics_ = ssrt_model.fit_transform(trialdata)
        return(self.metrics_)


if __name__ == '__main__':
    ssd = fixedSSD(np.arange(0, 550, 50))
    study = StopTaskStudy(ssd)
    trialdata = study.run()

    # summarize data - go trials are labeled with SSD of -inf so that
    # they get included in the summary
    print(trialdata.groupby('SSD').mean())
    print('go_accuracy', trialdata.query('trialtype=="go"').correct.mean())
    print(study.get_stopsignal_metrics())
