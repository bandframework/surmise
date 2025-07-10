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
    def lpdf(theta):
        return (sps.norm.logpdf(theta[:, 0], 0, 5) +
                sps.gamma.logpdf(theta[:, 1], 2, 0, 10)).reshape((len(theta), 1))

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


# tests for prediction class methods:
# test to check the prediction.mean()
@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
def test_prediction_mean(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.mean() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.var()
def test_prediction_var(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.var() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.covx()
def test_prediction_covx(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.covx() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.covxhalf()
def test_prediction_covxhalf(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.covxhalf() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', pytest.raises(ValueError)),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.mean_gradtheta()
def test_prediction_mean_gradtheta(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta, args={'return_grad': True})
    with expectation:
        assert pred.mean_gradtheta() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', pytest.raises(ValueError)),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.covx_gradtheta()
def test_prediction_covxhalf_gradtheta(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta, args={'return_grad': True})
    with expectation:
        assert pred.covxhalf_gradtheta() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', pytest.raises(ValueError))  # PCSK does not support remove function
    ],
    )
# test to check emulator.remove()
def test_remove(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)

    with expectation:
        emu.remove(theta=theta1)
        assert len(emu._emulator__theta) == 25, 'Check emulator.remove()'


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', pytest.raises(IndexError))  # PCSK does not support update function
    ],
    )
# test to check emulator.update()
def test_update(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    thetanew = priorphys_lin.rnd(10)
    fnew = balldropmodel_linear(xv, thetanew)
    with expectation:
        emu.update(x=None, theta=thetanew, f=fnew)
        assert len(emu._emulator__theta) == 60, 'Check emulator.update()'
