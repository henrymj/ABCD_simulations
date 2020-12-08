# class for a single run of a stop study (i.e. a single subject)

import argparse
import json
import os
import numpy as np
from trial import Trial, init_default_params, fix_params
from collections import namedtuple
import pandas as pd
from stopsignalmetrics import StopData, SSRTmodel
from ssd import trackingSSD, fixedSSD
Trialdata = namedtuple('Trialdata', 'trialtype, SSD, rt, resp, correct')


# main study class
class StopTaskStudy:
    def __init__(self, SSDgenerator, outdir, params=None, **kwargs):
        self.params = init_default_params() if params is None else params
        for key, value in kwargs.items():
            self.params[key] = value

        self.params = fix_params(self.params)
        self.outdir = outdir

        if outdir is not None and not os.path.exists(outdir):
            os.makedirs(outdir)
        self.SSDgenerator = SSDgenerator
        self.trialdata_ = None
        self.metrics_ = None

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
        for _ in range(self.params['ntrials']['stop']):
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

        # generate stop trials depending on the generator mechanism
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
        trialdata.loc[trialdata.correct.isnull(), 'correct'] = False
        trialdata['correct_response'] = 1
        var_dict = {
            'columns': {'ID': 'subcode',
                        'block': 'block',
                        'condition': 'trialtype',
                        'SSD': 'SSD',
                        'goRT': 'rt',
                        'stopRT': 'rt',
                        'response': 'resp',
                        'correct_response': 'correct_response',
                        'choice_accuracy': 'correct'},
            'key_codes': {'go': 'go',
                          'stop': 'stop',
                          'correct': True,
                          'incorrect': False,
                          'noResponse': None}}
        stopdata = StopData(var_dict, compute_acc_col=False)
        trialdata_proc = stopdata.fit_transform(trialdata)
        # compute metrics
        ssrt_model = SSRTmodel(model='all')
        self.metrics_ = ssrt_model.fit_transform(trialdata_proc)
        return(self.metrics_)

    def save_metrics(self):
        if self.metrics_ is None:
            raise Exception('metrics not yet computed')
        outfile_stem = os.path.join(self.outdir, f'{self.params["subcode"]}')
        with open(f'{outfile_stem}_metrics.json', 'w') as f:
            json.dump(self.metrics_, f)


def get_args():
    parser = argparse.ArgumentParser(description='ABCD data simulator')
    parser.add_argument('--paramfile', help='json file containing parameters')
    parser.add_argument('--min_ssd', help='minimum SSD value', default=0)
    parser.add_argument('--max_ssd', help='maximum SSD value', default=550)
    parser.add_argument('--ssd_step', help='SSD step size', default=50)
    parser.add_argument('--random_seed', help='random seed', type=int)
    parser.add_argument('--p_guess_go', help='p_guess on go trials (default None)')
    parser.add_argument('--stop_guess_ABCD', help='use SSD-dependent guessing based on ABCD data', action='store_true')

    parser.add_argument('--guess_param_file', default='exgauss_params.json',
                        help='file with exgauss params for guesses')
    parser.add_argument('--tracking', help='use tracking algorithm', action='store_true')
    parser.add_argument('--n_subjects', type=int,
                        help='number of subjects to simulate', default=1)
    parser.add_argument('--out_dir',
                        default='./simulated_data/pseudosubjects',
                        help='location to save simulated data')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(f'simulating stop task for {args.n_subjects} subjects')
    if args.paramfile is not None:
        with open(args.paramfile) as f:
            params = json.load(f)
        if 'p_guess' not in params:
            params['p_guess'] = {'go': None, 'stop': None}
        if args.p_guess_go is not None:
            params['p_guess']['go'] = float(args.p_guess_go)
        if args.stop_guess_ABCD is not None:
            params['p_guess']['stop'] = 'ABCD'
    else:
        params = None
    print(params)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    if args.tracking:
        ssd = trackingSSD()
    else:
        ssd = fixedSSD(np.arange(args.min_ssd, args.max_ssd + args.ssd_step, args.ssd_step))

    for i in range(args.n_subjects):
        print(f'running subject {i + 1}')
        study = StopTaskStudy(ssd, args.out_dir, params=params)

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
        study.save_metrics()
