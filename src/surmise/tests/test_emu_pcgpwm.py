import numpy as np
import scipy.stats as sps
import pytest
from surmise.emulation import emulator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f, _datagen

pytestmark = pytest.mark.usefixtures('seeded_rng')

##############################################
#            Simple scenarios                #
##############################################
f1 = f[0:15, :]
f2 = f[:, 0:25]
theta1 = theta[0:25, :]
x1 = x[0:15, :]
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)
simsd = 1e-3 * np.ones_like(f)


##############################################
# Unit tests to initialize an emulator class #
##############################################
# missing mask
maskU = sps.uniform.rvs(size=f.size, random_state=_datagen).reshape(*f.shape)
# tests missing data
f_miss = f.copy()
f_miss[maskU < 0.2] = np.nan


@pytest.mark.parametrize(
    "input1, expectation",
    [
     (f, does_not_raise()),
     (f_miss, does_not_raise()),
     ],
    )
def test_fmissing(input1, expectation):
    with expectation:
        assert emulator(x=x, theta=theta, f=input1,
                        method='PCGPwM') is not None


U, S, _ = np.linalg.svd(f, full_matrices=False)
pcinfo = {'U': U}


@pytest.mark.parametrize("input1, expectation",
                         [({}, pytest.raises(AttributeError)),
                          (pcinfo, does_not_raise())])
def test_supply_pcinfo(input1, expectation):
    with expectation:
        assert emulator(x=x, theta=theta, f=f,
                        method='PCGPwM',
                        args={'standardpcinfo': input1}) is not None


# test to check the prediction.mean_gradtheta()
@pytest.mark.parametrize(
    "input1,input2,expectation",
    [
     ('PCGPwM', False, pytest.raises(ValueError)),
     ('PCGPwM', True, does_not_raise()),
     ],
    )
def test_prediction_mean_gradtheta(input1, input2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta, args={'return_grad': input2})
    with expectation:
        assert pred.mean_gradtheta() is not None


# test to check the prediction.covxhalf_gradtheta()
@pytest.mark.parametrize(
    "input1,return_grad, return_covx,traint,testt,expectation",
    [
     ('PCGPwM', False, False, theta, theta, pytest.raises(ValueError)),
     ('PCGPwM', True, False, theta, theta, pytest.raises(ValueError)),
     ('PCGPwM', True, True, theta, theta, does_not_raise()),
     ('PCGPwM', True, False, theta, theta1, pytest.raises(ValueError)),
     ('PCGPwM', True, True, theta, theta1, does_not_raise()),
     ],
    )
def test_prediction_covxhalf_gradtheta(input1, return_grad, return_covx,
                                       traint, testt, expectation):
    emu = emulator(x=x, theta=traint, f=f, method=input1)
    pred = emu.predict(x=x, theta=testt, args={'return_covx': return_covx,
                                               'return_grad': return_grad})
    with expectation:
        assert pred.covxhalf_gradtheta() is not None


@pytest.mark.parametrize(
    "verbose,expectation",
    [
     (0, does_not_raise()),
     (1, does_not_raise()),
     (2, does_not_raise()),
     ],
    )
def test_verbosity(verbose, expectation):
    with expectation:
        assert emulator(x=x, theta=theta, f=f, method='PCGPwM',
                        args={'verbose': verbose}) is not None
