import numpy as np
import scipy.stats as sps
import pytest
from contextlib import contextmanager
from surmise.emulation import emulator
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

##############################################
#            Simple scenarios                #
##############################################
def balldropmodel_linear(x, theta):
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1] + theta[k, 0]
        vter = theta[k, 1]
        f[k, :] = h0 - vter * t
    return f.T


tvec = np.concatenate((np.arange(0.1, 4.3, 0.1), np.arange(0.1, 4.3, 0.1)))
h0vec = np.concatenate((25 * np.ones(42), 50 * np.ones(42)))
x = np.array([[0.1, 25.],
              [0.2, 25.],
              [0.3, 25.],
              [0.4, 25.],
              [0.5, 25.],
              [0.6, 25.],
              [0.7, 25.],
              [0.9, 25.],
              [1.1, 25.],
              [1.3, 25.],
              [2.0, 25.],
              [2.4, 25.],
              [0.1, 50.],
              [0.2, 50.],
              [0.3, 50.],
              [0.4, 50.],
              [0.5, 50.],
              [0.6, 50.],
              [0.7, 50.],
              [0.8, 50.],
              [0.9, 50.],
              [1.0, 50.],
              [1.2, 50.],
              [2.6, 50.],
              [2.9, 50.],
              [3.1, 50.],
              [3.3, 50.],
              [3.5, 50.],
              [3.7, 50.], ]).astype('object')
xv = x.astype('float')


class priorphys_lin:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.norm.logpdf(theta[:, 0], 0, 5) +
                              sps.gamma.logpdf(theta[:, 1], 2, 0, 10))
        else:
            return np.squeeze(sps.norm.logpdf(theta[0], 0, 5) +
                              sps.gamma.logpdf(theta[1], 2, 0, 10))

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n),
                          sps.gamma.rvs(2, 0, 10, size=n))).T


theta = priorphys_lin.rnd(50)
f = balldropmodel_linear(xv, theta)
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
theta1d = sps.norm.rvs(0, 5, size=50)
# 1-d theta
theta1dx = sps.norm.rvs(0, 5, size=2)
##############################################
# Unit tests to initialize an emulator class #
##############################################


@contextmanager
def does_not_raise():
    yield


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


# test to check the predict call()
@pytest.mark.parametrize(
    "expectation",
    [
     (pytest.raises(ValueError)),  # rnd is not in the method
     ],
    )
def test_predict_call(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    emu_pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert emu_pred(s=10) is not None


# test to check the prediction.mean()
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ('PCGP', does_not_raise()),
     ],
    )
def test_prediction_mean(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.mean() is not None


# test to check the prediction.mean_gradtheta()
# @pytest.mark.parametrize(
#     "input1,input2,expectation",
#     [
#      ('PCGP', False, pytest.raises(ValueError)),
#      ('PCGP', True, does_not_raise()),
#      ],
#     )
# def test_prediction_mean_gradtheta(input1, input2, expectation):
#     emu = emulator(x=x, theta=theta, f=f, method=input1)
#     pred = emu.predict(x=x, theta=theta, args={'return_grad': input2})
#     with expectation:
#         assert pred.mean_gradtheta() is not None


# test to check the prediction.var()
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ('PCGPwM', does_not_raise()),
     ('PCGP', does_not_raise()),
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
# @pytest.mark.parametrize(
#     "input1,input2,expectation",
#     [
#      ('PCGP', False, pytest.raises(ValueError)),
#      ('PCGP', True, does_not_raise()),
#      ],
#     )
# def test_prediction_covxhalf_gradtheta(input1, input2, expectation):
#     emu = emulator(x=x, theta=theta, f=f, method=input1)
#     pred = emu.predict(x=x, theta=theta, args={'return_grad': input2})
#     with expectation:
#         assert pred.covxhalf_gradtheta() is not None


# test to check the prediction.rnd()
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ('PCGP', pytest.raises(ValueError)),
     ],
    )
def test_prediction_rnd(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.rnd() is not None


# test to check the prediction.lpdf()
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ('PCGP', pytest.raises(ValueError)),
     ],
    )
def test_prediction_lpdf(input1, expectation):
    emu = emulator(x=x, theta=theta, f=f, method=input1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.lpdf() is not None
