# test stoptaskstudy functions

from stoptaskstudy import fixedSSD, StopTaskStudy

def test_fixedssd():
    SSDvals = [0, 50, 100, 150, 200]
    ssd = fixedSSD(SSDvals)
    ssds = [i for i in ssd.generate(20)]
    assert len(set(SSDvals).difference(set(ssds))) == 0


def test_run_fixedssd():
    ssd = fixedSSD([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    study = StopTaskStudy(ssd)
    trialdata = study.run()
