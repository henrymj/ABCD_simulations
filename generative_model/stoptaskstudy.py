# class for a single run of a stop study (i.e. a single subject)

import argparse
import json
import os
import numpy as np
from trial import Trial
from collections import namedtuple
import pandas as pd
from stopsignalmetrics import StopData, SSRTmodel, PostStopSlow, Violations, StopSummary

Trialdata = namedtuple('Trialdata', 'trialtype, SSD, rt, resp, correct')

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


# based on the ABCD tracking algorithm
class trackingSSD:
    def __init__(self, starting_ssd=50, step_size=50, min_ssd=0, max_ssd=550):
        self.starting_ssd = starting_ssd
        self.step_size = step_size
        self.min_ssd = min_ssd
        self.max_ssd = max_ssd

        self.SSD = starting_ssd

    def update(self, SSD, success):
        """[summary]

        Args:
            SSD (int): current sssd
            success (boolean): success or failure

        Returns:
            SSD (int): updated SSD
        """
        if SSD is None:
            SSD = self.starting_ssd
        elif success == True:
            SSD = SSD + self.step_size
        else:
            SSD = SSD - self.step_size

        if SSD < self.min_ssd:
            self.SSD = min_ssd
        elif SSD > self.max_ssd:
            self.SSD = max_ssd
        else:
            self.SSD = SSD

        return(self.SSD)

# main study class
class StopTaskStudy:
    def __init__(self, SSDgenerator, outdir, params=None, **kwargs):
        if params is None:
            self.params = {
                'mu': {'go': 0.3, 'stop': 0.5},
                'max_time': 1000,
                'mu_delta_incorrect': 0.2,
                'nondecision': {'go': 50, 'stop': 50},
                'noise_sd': {'go': 2.2, 'stop': 2.2},
                'threshold': 100,
                'ntrials': {'go': 10000, 'stop': 2000},
                'mu_go_grader': 'log'
            }
        else:
            self.params = params
        for key, value in kwargs.items():
            self.params[key] = value
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.SSDgenerator = SSDgenerator
        self.trialdata_ = None
        # create random subject code
        if 'subcode' not in self.params:
            self.params['subcode'] = np.random.randint(10e12)

    def _generate_stop_trials_fixed(self):
        # generate stop trials
        trialdata = []
        SSD = None
        for SSD in self.SSDgenerator.generate(self.params['ntrials']['stop']):
            trial = Trial(SSD, self.params)
            rt, correct = trial.simulate()
            trialdata.append(Trialdata(
                trialtype='stop',
                SSD=SSD,
                rt=rt,
                resp=(rt is not None),
                correct=correct))
        return(trialdata)

    def _generate_stop_trials_tracking(self):
        # generate stop trials
        trialdata = []
        SSD = None
        stop_success = None
        for i in range(self.params['ntrials']['stop']):
            SSD = self.SSDgenerator.update(SSD, stop_success)
            trial = Trial(SSD, self.params)
            rt, correct = trial.simulate()
            trialdata.append(Trialdata(
                trialtype='stop',
                SSD=SSD,
                rt=rt,
                resp=(rt is not None),
                correct=correct))
            stop_success = rt is None
        return(trialdata)

    def run(self):

        if self.SSDgenerator.__class__.__name__ == 'fixedSSD':
            trialdata = self._generate_stop_trials_fixed()
        elif self.SSDgenerator.__class__.__name__ == 'trackingSSD':
            trialdata = self._generate_stop_trials_tracking()
        else:
            raise Exception(f'SSD generator class {self.SSDgenerator.__class__.__name__} not yet implemented')
 


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

    def save_trialdata(self, save_params=True):
        if self.trialdata_ is None:
            raise Exception('Trialdata not yet computed')
        outfile_stem = os.path.join(self.outdir, f'{self.params["subcode"]}')
        self.trialdata_.to_csv(f'{outfile_stem}.csv')
        with open(f'{outfile_stem}.json', 'w') as f:
            json.dump(self.params, f)
        

    def get_stopsignal_metrics(self):
        trialdata = self.trialdata_
        # clean up data for stopsignal metrics analysis
        trialdata['subcode'] = self.params['subcode']
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


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulator')
    parser.add_argument('--paramfile', help='json file containing parameters')
    parser.add_argument('--min_ssd', help='minimum SSD value', default=0)
    parser.add_argument('--max_ssd', help='maximum SSD value', default=550)
    parser.add_argument('--ssd_step', help='SSD step size', default=50)
    parser.add_argument('--random_seed', help='random seed', type=int)
    parser.add_argument('--n_subjects', nargs='+',
                        help='number of subjects to simulate', default=1)
    parser.add_argument('--out_dir',
                        default='./simulated_data/pseudosubjects',
                        help='location to save simulated data')
    args = parser.parse_args()
    return(args)


if __name__ == '__main__':
    args = get_args()
    print(f'simulating stop task for {args.n_subjects} subjects')
    if args.paramfile is not None:
        with open(args.paramfile) as f:
            params = json.load(f)
    else:
        params = None
    print(params)
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    #ssd = fixedSSD(np.arange(args.min_ssd, args.max_ssd + args.ssd_step, args.ssd_step))
    ssd = trackingSSD()
    study = StopTaskStudy(ssd, args.out_dir)

    # save some extra params for output to json
    study.params['args'] = args.__dict__
    study.params['pwd'] = os.getcwd()

    trialdata = study.run()
    study.save_trialdata()

    # summarize data - go trials are labeled with SSD of -inf so that
    # they get included in the summary
    print(trialdata.groupby('SSD').mean())
    print('go_accuracy', trialdata.query('trialtype=="go"').correct.mean())
    print(study.get_stopsignal_metrics())
