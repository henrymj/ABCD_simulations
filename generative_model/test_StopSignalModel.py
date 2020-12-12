# tests for new stop signal model class

from pyABC import StopSignalModel
import pytest


def test_ssm_smoke():
    ssm = StopSignalModel('basic', parameters={'mugo': 2}, mustop=3)
    assert ssm is not None
    assert ssm.parameters['mugo'] == 2
    assert ssm.parameters['mustop'] == 3


# test all of the possible models
@pytest.mark.parametrize(
    "model",
    ['basic',
     'simpleguessing',
     'scaledguessing',
     'gradedmugo',
     'fullabcd',
     'cacaphony'])
def test_ssm_models(model):
    ssm = StopSignalModel(model, paramfile=f'params/params_{model}.json')
    assert ssm is not None
    modelparams = {
        'mu_delta_incorrect': 0.1083986991950653,
        'mu_go': 0.16778296718781244,
        'mu_stop_delta': 0.012833735851097903,
        'noise_sd': 2.511930980540796,
        'nondecision': 73.12196295189378}
    if model in ['simpleguessing', 'scaledguessing', 'fullabcd', 'cacaphony']:
        modelparams['pguess'] = 0.05
    result = ssm.fit_transform(modelparams)
    assert result is not None
    assert all(item in result for item in [
        'mean_go_RT', 'mean_stopfail_RT', 'go_acc', 'sd_go_RT', 'sd_stopfail_RT'])
