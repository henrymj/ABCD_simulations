from .simulate_generative import SimulateGenerative
import pytest

@pytest.fixture(scope="session")
def simulator():
    return(SimulateGenerative())

@pytest.fixture(scope="session")
def params():
    return({
        'mu_go': 0.5, 
        'mu_stop': 0.5,
        'n_trials': 100,
        'max_time': 1000,
        'nondecision_go': 50,
        'noise_go': 3.2,
        'threshold': 100
    })


def test_SimulateGenerative_smoke(simulator):
    assert simulator is not None

def test_simulate_accumumator_and_get_rt(simulator, params):
    # test with high mu that should trigger a response
    accum = simulator._simulate_accumulator(max_time=params['max_time'])
    assert len(accum) == params['max_time']
    rt = simulator._get_rt_from_accumulator(accum, params['threshold'])
    assert accum[rt] > params['threshold']

def test_simulate_accumumator_none(simulator, params):
    # test with zero drift and noise to ensure that it properly returns None
    accum = simulator._simulate_accumulator(starting_point=50, mu=0., 
                              noise_sd=0., max_time=1000)
    rt = simulator._get_rt_from_accumulator(accum, params['threshold'])
    assert rt is None
   