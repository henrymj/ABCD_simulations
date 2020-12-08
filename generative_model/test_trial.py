# test Trial class

import pytest
from trial import Trial


@pytest.fixture(scope="session")
def trial():
    return(Trial())


def test_trial_smoke(trial):
    assert trial is not None


def test_trial_simulate_basic(trial):
    rt, correct = trial.simulate()
    assert rt is not None
    assert correct is not None


# test kwargs, set mu/noise to go
# which results in no response
def test_trial_simulate_kwargs(trial):
    rt, correct = trial.simulate(
        mu={'go': 0, 'stop': 0},
        noise_sd={'go': 0, 'stop': 0},
        verbose=True)
    assert rt is None
    assert correct is None


# set mu_stop high and mu_go low to ensure successful stopping
def test_trial_simulate_stop(trial):
    rt, correct = trial.simulate(SSD=0, mu={'go': 0.1, 'stop': 0.9})
    assert rt is None  # should be a succesful stop
    assert correct is None


# test setting params at instantiation
def test_trial_kwargs():
    trial = Trial(mu={'go': 0.0, 'stop': 0.9}, noise_sd={'go': 0.1, 'stop': 0.1})
    rt, correct = trial.simulate(SSD=-100, verbose=True)
    assert rt is None  # should be a succesful stop
    assert correct is None


# test with guessing

def test_guess_func():
    # go trial, go guessing turned off, should return None
    p_guess = {'go': None, 'stop': None}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(p_guess=p_guess)
    rt, correct = trial.get_guess_rt(trial.params)
    assert rt is None and correct is None

    # go trial, go guessing turned one, should return non-None
    p_guess = {'go': 1., 'stop': None}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(p_guess=p_guess)
    rt, correct = trial.get_guess_rt(trial.params)
    assert rt is not None and correct is not None

    # stop trial, stop guessing turned off, should return None
    p_guess = {'go': None, 'stop': None}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(SSD=0, p_guess=p_guess)
    rt, correct = trial.get_guess_rt(trial.params)
    assert rt is None and correct is None

    # stop trial, stop guessing turned on, should return non-None
    p_guess = {'go': None, 'stop': 1}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(SSD=0, p_guess=p_guess)
    rt, correct = trial.get_guess_rt(trial.params)
    assert rt is not None and correct is not None

    # stop trial, stop guessing turned on, should return non-None
    p_guess = {'go': None, 'stop': 'ABCD'}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(SSD=0, p_guess=p_guess)
    rt, correct = trial.get_guess_rt(trial.params)
    assert rt is not None and correct is not None

    # stop trial, stop guessing turned on, should return non-None
    p_guess = {'go': None, 'stop': 'ABCD'}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(SSD=550, p_guess=p_guess, mu=mu)
    rt, correct = trial.get_guess_rt(trial.params)
    assert rt is None and correct is None


def test_trial_pguess_go():
    p_guess = {'go': 1., 'stop': None}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(p_guess=p_guess, mu=mu)
    rt, correct = trial.simulate()
    assert trial.params['is_guess']


def test_trial_pguess_false():
    p_guess = {'go': 0., 'stop': None}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(p_guess=p_guess, mu=mu)
    rt, correct = trial.simulate()
    assert trial.params['is_guess'] is False


def test_trial_pguess_stop():
    p_guess = {'go': 1., 'stop': 'ABCD'}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(SSD=0, p_guess=p_guess, mu=mu)
    rt, correct = trial.simulate()
    assert trial.params['is_guess']


def test_trial_pguess_stop_longSSD():
    p_guess = {'go': 1., 'stop': 'ABCD'}
    mu = {'go': 0.5, 'stop': 0.9}
    trial = Trial(SSD=550, p_guess=p_guess, mu=mu)
    rt, correct = trial.simulate()
    assert trial.params['is_guess'] is False
