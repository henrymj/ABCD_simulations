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
