import numpy as np
import scipy.stats as sps
import pytest
from surmise.emulation import emulator
from surmise.calibration import calibrator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f, y_lin as y, \
                             obsvar_lin as obsvar, priorphys_lin, RNG_SEED, DEFAULT_MH_SPECS

pytestmark = pytest.mark.usefixtures('seeded_rng')

##############################################
#            Simple scenarios                #
##############################################
f1 = f[0:15, :]
f2 = f[:, 0:25]
theta1 = theta[0:25, :]
x1 = x[0:15, :]
x1d = x[:, 0].reshape((x.shape[0],))
theta4d = np.hstack((theta1, theta1))
# Do not use surmise RNG outside of the tests
_rng = np.random.default_rng(RNG_SEED)
thetarnd = np.vstack((sps.norm.rvs(0, 5, size=20, random_state=_rng),
                      sps.gamma.rvs(2, 0, 10, size=20, random_state=_rng))).T
thetarnd2 = np.vstack((sps.norm.rvs(0, 5, size=10, random_state=_rng),
                       sps.gamma.rvs(2, 0, 10, size=10, random_state=_rng))).T
thetacomb = np.vstack((theta1, thetarnd))


#######################################################
# Unit tests for supplement() method of emulator class #
#######################################################
# test to check supplement_x
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
        (5, x, x1, pytest.raises(ValueError)),  # not supported
        (0.25, x, x1, pytest.raises(ValueError)),  # must be integer
        (5, None, x1, pytest.raises(ValueError)),
    ],
)
def test_supplement_x(input1, input2, input3, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    with expectation:
        assert emu.supplement(size=input1,
                              x=input2,
                              xchoices=input3) is not None


# test to check supplement_theta
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
        # replication of emu.__theta
        (0, theta, theta1, pytest.raises(ValueError)),  # 'No supptheta exists.'
        (5, theta, theta1, pytest.raises(ValueError)),
        # 'Complete replication of self.__theta'
        (5, None, theta1, pytest.raises(ValueError)),
        # 'Provide either x or (theta or cal).'
        (5, theta, theta4d, pytest.raises(ValueError)),  # 'Dimension.'
        (5, theta, None, pytest.raises(ValueError)),
        # 'Complete replication of self.__theta'
        (5, theta4d, None, pytest.raises(ValueError)),
        # (5, thetarnd, None, does_not_raise()),
        (5, thetacomb, None, pytest.raises(ValueError)),
    ],
)
def test_supplement_theta(input1, input2, input3, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    with expectation:
        assert emu.supplement(size=input1,
                              theta=input2,
                              thetachoices=input3) is not None


# test to check supplement_theta pending argument
@pytest.mark.parametrize(
    "includepending,expectation",
    [
        (True, does_not_raise()),
        (False, does_not_raise()),
    ],
)
def test_supplement_pending(includepending, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    with expectation:
        assert emu.supplement(size=5,
                              theta=thetarnd,
                              thetachoices=thetarnd2[:5],
                              args={'pending': thetarnd2[5:],
                                    'includepending': includepending}) is not None


# test to check supplement_theta
@pytest.mark.parametrize(
    "input1,input2,expectation",
    [
        (x, theta, pytest.raises(ValueError)),
        # ValueError: You must either provide either x or (theta or cal).
        (None, None, pytest.raises(ValueError)),
        # ValueError: You must either provide either x or (theta or cal).
    ],
)
def test_supplement_x_theta(input1, input2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    with expectation:
        assert emu.supplement(size=10, x=input1, theta=input2) is not None


# test to check supplement_cal
@pytest.mark.parametrize(
    "expectation",
    [
        (does_not_raise()),
    ],
)
def test_supplement_cal(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=priorphys_lin,
                     method='directbayes',
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS)
    with expectation:
        assert emu.supplement(size=10, cal=cal) is not None


# test to check supplement_cal
@pytest.mark.parametrize(
    "expectation",
    [
        (does_not_raise()),
    ],
)
def test_supplement_supp(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    emu.supplement(size=5, theta=thetarnd)

    with expectation:
        assert emu.supplement(size=0) is not None


# test to check supplement_cal
@pytest.mark.parametrize(
    "expectation",
    [
        (pytest.raises(ValueError)),
    ],
)
def test_supplement_method(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu.supplement(size=5, theta=thetarnd) is not None

# test to check supplement_theta
# @pytest.mark.parametrize(
#    "input1,expectation",
#    [
#    (thetacomb, does_not_raise()),
#       ValueError: You must either provide either x or (theta or cal).
#    ],
#    )
# def test_supplement_match(input1, expectation):
#    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
#    with expectation:
#        assert emu.supplement(size=15, theta=theta, thetachoices=input1) is not None
