# test stoptaskstudy functions

from stoptaskstudy import fixedSSD, trackingSSD, StopTaskStudy
import pytest

@pytest.fixture(scope="session")
def study_fixed():
    ssd = fixedSSD([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    return(StopTaskStudy(ssd, '/dev/null'))


@pytest.fixture(scope="session")
def study_tracking():
    ssd = trackingSSD()
    return(StopTaskStudy(ssd, '/dev/null'))


def test_fixedssd():
    SSDvals = [0, 50, 100, 150, 200]
    ssd = fixedSSD(SSDvals)
    ssds = [i for i in ssd.generate(20)]
    assert len(set(SSDvals).difference(set(ssds))) == 0


def test_run_fixedssd(study_fixed):
    trialdata = study_fixed.run()
    assert trialdata is not None
    study_fixed.get_stopsignal_metrics()
    assert study_fixed.metrics_ is not None


def test_run_tracking(study_tracking):
    trialdata = study_tracking.run()
    assert trialdata is not None
    study_tracking.get_stopsignal_metrics()
    assert study_tracking.metrics_ is not None
