import os

##############################################
#            Simple scenarios                #
##############################################
import numpy as np
import pytest
from surmise.emulation import emulator
from surmise.calibration import calibrator
from .conftest import does_not_raise
from .shared_scenario import y_td as y, obsvar_td as obsvar, \
    x_std, theta_ball as theta, x_range, theta_range, prior_balldrop, timedrop

pytestmark = pytest.mark.usefixtures('seeded_rng')

##############################################
#            Simple scenarios                #
##############################################
# Obtain computer model output via filtered data
f = timedrop(x_std, theta, x_range, theta_range)

# Fit an emulator via non-filtered data
emulator_nf_1 = emulator(x=x_std, theta=theta, f=f, method='PCGP')
pred_nf = emulator_nf_1.predict(x=x_std, theta=theta)
pred_nf_mean = pred_nf.mean()

# Filter out the data
ys = 1 - np.sum((pred_nf_mean - y)**2, 0)/np.sum((y - np.mean(y))**2, 0)
theta_f = theta[ys > 0.5]

# Obtain computer model output via filtered data
f_f = timedrop(x_std, theta_f, x_range, theta_range)

# Fit an emulator via filtered data
emulator_f_1 = emulator(x=x_std, theta=theta_f, f=f_f, method='PCGP')

##############################################
# Unit tests to initialize an emulator class #
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
def test_cal_saveload(load_cal_flag, expectation):
    with expectation:
        cal = calibrator(emu=emulator_f_1,
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
def test_calpred_saveload(expectation):
    with expectation:
        cal = calibrator(emu=emulator_f_1,
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
