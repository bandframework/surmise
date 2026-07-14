import pytest
from surmise.emulation import emulator
from surmise.calibration import calibrator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f, y_lin as y, \
                             obsvar_lin as obsvar, priorphys_lin

pytestmark = pytest.mark.usefixtures('seeded_rng')


##############################################
#            Simple scenarios                #
##############################################
theta1 = theta[0:25, :]


#######################################################
# Unit tests for remove method of emulator class #
#######################################################
# test to check remove
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (theta1, does_not_raise()),
     ],
    )
def test_remove(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu.remove(theta=input1) is None


# test to check remove with a calibrator
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (theta1, does_not_raise()),
     ],
    )
def test_remove_cal(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    cal_bayes = calibrator(emu=emu,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method='directbayeswoodbury',
                           yvar=obsvar)
    with expectation:
        assert emu.remove(theta=input1, cal=cal_bayes) is None
