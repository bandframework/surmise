import numpy as np
import scipy.stats as sps
import pytest
from contextlib import contextmanager
from surmise.emulation import emulator

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
    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n),
                          sps.gamma.rvs(2, 0, 10, size=n))).T


theta = priorphys_lin.rnd(50)
f = balldropmodel_linear(xv, theta)
f1 = f[0:15, :]
f2 = f[:, 0:25]
theta1 = theta[0:25, :]
x1 = x[0:15, :]
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)
simsd = 1e-3 * np.ones_like(f)


def balldroptrue(x):
    def logcosh(x):
        # preventing crashing
        s = np.sign(x) * x
        p = np.exp(-2 * s)
        return s + np.log1p(p) - np.log(2)
    t = x[:, 0]
    h0 = x[:, 1]
    vter = 20
    g = 9.81
    y = h0 - (vter ** 2) / g * logcosh(g * t / vter)
    return y


obsvar = 4*np.ones(x.shape[0])
y = balldroptrue(xv)
##############################################
# Unit tests to initialize an emulator class #
##############################################


@contextmanager
def does_not_raise():
    yield


# tests missing data
f_miss = f.copy()
f_miss[np.random.rand(*f.shape) < 0.2] = np.nan


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
