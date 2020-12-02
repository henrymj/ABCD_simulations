# tests for accumulator class

import pytest
from .accumulator import Accumulator

@pytest.fixture(scope="session")
def accumulator():
    return(Accumulator(mu=0.5, noise_sd=0.3, starting_point=50, max_time=1000))

def test_accumulator_class_smoke(accumulator):
    assert accumulator is not None

def test_accumulator_run(accumulator):
    accumulator.run()
    assert len(accumulator.accum_) == accumulator.max_time

def test_threshold_accumulator(accumulator):
    accumulator.run()
    rt = accumulator.threshold_accumulator(100)
    assert rt is not None
    assert rt == accumulator.rt_
    

def test_threshold_accumulator_noise():
    # for negative mu, correct should be false
    accumulator = Accumulator(mu=0, noise_sd=0.01, starting_point=50, max_time=1000)
    accumulator.run()
    rt = accumulator.threshold_accumulator(100)
    assert rt is  None

def test_threshold_alone():
    accumulator = Accumulator(mu=0.5, noise_sd=0.3, starting_point=50, max_time=1000)
    rt = accumulator.threshold_accumulator(100)
    assert rt is not None

