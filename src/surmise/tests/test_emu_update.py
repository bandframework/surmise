import numpy as np
import pytest
from surmise.emulation import emulator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f, \
    f_new_lin as f_new, theta_new_lin as theta_new

pytestmark = pytest.mark.usefixtures('seeded_rng')

##############################################
#            Simple scenarios                #
##############################################
f1 = f[0:15, :]
x1 = x[0:15, :]

f1theta = f[:, 0:15]
theta1 = theta[0:15, :]

fmatch = np.hstack((f1theta, f_new))
thetamatch = np.vstack((theta1, theta_new))

x_new = x[15:30, :] + 1
xmatch = np.vstack((x1, x_new))
fmatchx = np.vstack((f1, f[15:30, :]))

fd = np.hstack((f, f))
#######################################################
# Unit tests for update method of emulator class #
#######################################################


# test to check update(): 'xreps'
@pytest.mark.parametrize(
    "input1,input2,input3,input4,expectation",
    [
     (x, theta, f, False, pytest.raises(ValueError)),
     (x, None, f, False, does_not_raise()),
     (x, None, f, True, does_not_raise()),
     (x1, None, f, True, pytest.raises(ValueError)),
     (x, None, f1, True, pytest.raises(ValueError)),
     (x1, None, f1, True, does_not_raise()),
     (xmatch, None, fmatchx, False, does_not_raise()),
     ],
    )
def test_update_x(input1, input2, input3, input4, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu.update(x=input1,
                          theta=input2,
                          f=input3,
                          options={'xreps': input4}) is None


# test to check update(): 'thetareps'
@pytest.mark.parametrize(
    "input1,input2,input3,input4,expectation",
    [
     (x, theta, f, False, pytest.raises(ValueError)),
     (None, theta, f, False, does_not_raise()),
     (None, theta, f, True, does_not_raise()),
     (None, theta1, f, True, pytest.raises(ValueError)),
     (None, theta, f1theta, True, pytest.raises(ValueError)),
     (None, theta1, f1theta, True, does_not_raise()),
     (None, thetamatch, fmatch, False, does_not_raise()),
     (None, None, fd, False, pytest.raises(ValueError)),
     ],
    )
def test_update_theta(input1, input2, input3, input4, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu.update(x=input1,
                          theta=input2,
                          f=input3,
                          options={'thetareps': input4}) is None


# test to check update() with None
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
     (None, None, f, does_not_raise()),
     (None, theta, None, does_not_raise()),
     (x, None, None, does_not_raise()),
     (x1, None, None, pytest.raises(ValueError)),
     (None, theta1, None, pytest.raises(ValueError)),
     (None, None, f1, pytest.raises(ValueError)),
     ],
    )
def test_update_f(input1, input2, input3, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu.update(x=input1, theta=input2, f=input3) is None


# # test to check update() with None
@pytest.mark.parametrize(
    "input1,input2,expectation",
    [
      (f_new, True, does_not_raise()),
      (f_new, False, does_not_raise()),
      (np.hstack((f_new, f_new)), False, pytest.raises(ValueError)),
      ],
    )
def test_update_supptheta(input1, input2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    emu.supplement(size=10, theta=theta_new)
    with expectation:
        assert emu.update(f=input1, options={'thetareps': input2}) is None
