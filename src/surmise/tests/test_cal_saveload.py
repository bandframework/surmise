import os

##############################################
#            Simple scenarios                #
##############################################
import numpy as np
import pytest
from surmise.calibration import calibrator
from .conftest import does_not_raise
from .shared_scenario import y_td as y, obsvar_td as obsvar, \
    x_std, prior_balldrop

pytestmark = pytest.mark.usefixtures('seeded_rng', '_session_rng')


##############################################
#            Simple scenarios                #
##############################################
args2 = {'theta0': np.array([[0.4]]),
         'numsamp': 20,
         'stepType': 'normal',
         'stepParam': [0.4]}


@pytest.mark.parametrize(
    "load_cal_flag, expectation",
    [
     (True, does_not_raise()),
     (False, pytest.raises(TypeError))
     ],
    )
def test_cal_saveload(load_cal_flag, emu_timedrop, expectation):
    with expectation:
        cal = calibrator(emu=emu_timedrop,
                         y=y,
                         x=x_std,
                         thetaprior=prior_balldrop,
                         method='directbayes',
                         yvar=obsvar,
                         args=args2)

        fname = 'test_cal_saveload.pkl'
        cal.save_to(fname)

        if load_cal_flag:
            calload = calibrator.load_from(fname)
        else:
            try:
                calload = calibrator.load_prediction(fname)
            except TypeError:
                # in case test fails, generated files should be cleaned up
                os.remove(fname)
                raise TypeError
        assert calload.theta.mean() == cal.theta.mean()
        os.remove(fname)


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_calpred_saveload(emu_timedrop, expectation):
    with expectation:
        cal = calibrator(emu=emu_timedrop,
                         y=y,
                         x=x_std,
                         thetaprior=prior_balldrop,
                         method='directbayes',
                         yvar=obsvar,
                         args=args2)
        calpred = cal.predict(x=x_std)

        fname = 'test_calpred_saveload.pkl'
        calpred.save_to(fname)

        calpredload = calibrator.load_prediction(fname)
        assert (calpredload.mean() == calpred.mean()).all()
        os.remove(fname)
