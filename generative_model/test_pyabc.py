from pyABC import stopsignal_model_basic


def test_stopsignal_model():
    model_parameters = {'mu_delta_incorrect': 0.10386248711279868,
                        'mu_go': 0.11422675271126799,
                        'mu_stop_delta': 0.7850423871897488,
                        'noise_sd': 3.1238287051634597,
                        'nondecision': 50}

    simulation = stopsignal_model_basic(model_parameters)
    assert simulation is not None
