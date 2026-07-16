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

x1obs = x[0, :]
x1nothing = np.array([1, 2, 3])

f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)

x3 = np.vstack((np.array(list(np.arange(0, 10)) * 2),
                np.repeat([1, 2], 10), np.repeat([2, 3], 10))).T

# 1-d theta
theta1d = sps.norm.rvs(0, 5, size=50, random_state=_datagen)
# 1-d theta
theta1dx = sps.norm.rvs(0, 5, size=2, random_state=_datagen)
##############################################
# Unit tests to initialize an emulator class #
##############################################


# test to check the predict method with multivariate example
@pytest.mark.parametrize(
    "input1,input2,expectation",
    [
     (x, theta, does_not_raise()),
     (x.T, theta, does_not_raise()),
     (x1obs, theta, does_not_raise()),
     (x1nothing, theta, pytest.raises(ValueError)),
     (None, theta, does_not_raise()),
     (x3, theta, pytest.raises(ValueError)),
     (x, None, does_not_raise()),
     (x, theta.T, does_not_raise()),
     (x, theta1d, pytest.raises(ValueError)),
     (x, theta1dx, does_not_raise()),
     ],
    )
def test_predict_multi(input1, input2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu.predict(x=input1, theta=input2) is not None


# test to check the predict repr()
@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_predict_repr(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    emu_pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert repr(emu_pred) is not None


# test to check the prediction.mean()
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ('PCGPwM', does_not_raise()),
     ('PCGP', does_not_raise()),
     ('indGP', does_not_raise()),
     ('PCGPwImpute', does_not_raise()),
     ],
    )
def test_prediction_mean(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.mean() is not None


# test to check the prediction.var()
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ('PCGPwM', does_not_raise()),
     ('PCGP', does_not_raise()),
     ('indGP', does_not_raise()),
     ('PCGPwImpute', does_not_raise()),
     ],
    )
def test_prediction_var(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.var() is not None


# test to check the prediction.covx()
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ('PCGPwM', does_not_raise()),
     ('PCGP', does_not_raise()),
     ('indGP', does_not_raise())
     ],
    )
def test_prediction_covx(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.covx() is not None


# test to check the prediction.covxhalf()
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ('PCGPwM', does_not_raise()),
     ('PCGP', does_not_raise()),
     ],
    )
def test_prediction_covxhalf(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.covxhalf() is not None


# test to check the prediction.covxhalf_gradtheta()
@pytest.mark.parametrize(
    "input1, expectation",
    [
     ('PCGPwM', does_not_raise()),
     ('PCGP', pytest.raises(ValueError)),
     ],
    )
def test_prediction_covxhalf_gradtheta(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta, args={'return_grad': True})
    with expectation:
        assert pred.covxhalf_gradtheta() is not None


# test to check the prediction.lpdf()
@pytest.mark.parametrize(
    "input1, return_grad, expectation",
    [
     ('PCGP', False, pytest.raises(ValueError)),
     ('PCGPwM', True, does_not_raise()),
     ('PCGPwM', False, does_not_raise()),
     ('PCGPwImpute', True, does_not_raise()),
     ('PCGPwImpute', False, does_not_raise()),
     ],
    )
def test_prediction_lpdf(input1, return_grad, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta, args={'return_grad': return_grad})
    with expectation:
        assert pred.lpdf(f=f) is not None
