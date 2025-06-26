import numpy as np
import scipy.stats as sps
from contextlib import contextmanager
from surmise.emulation import emulator
import pytest
import os

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
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)

##############################################
# Unit tests to initialize an emulator class #
##############################################


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "load_emu_flag, expectation",
    [
     (True, does_not_raise()),
     (False, pytest.raises(TypeError))
     ],
    )
def test_emu_saveload(load_emu_flag, expectation):
    fname = 'test_emu_saveload.pkl'
    with expectation:
        emu = emulator(x=x, theta=theta, f=f)
        emu.save_to(fname)

        if load_emu_flag:
            emuload = emulator.load_from(fname)
        else:
            try:
                emuload = emulator.load_prediction(fname)
            except TypeError:
                # in case test fails, generated files should be cleaned up
                os.remove(fname)
                raise TypeError
        assert emuload is not None
        os.remove(fname)


def test_emupred_saveload():
    fname = 'test_emupred_saveload.pkl'
    with does_not_raise():
        emu = emulator(x=x, theta=theta, f=f)

        emupred = emu.predict()
        emupred.save_to(fname)

        emupredload = emulator.load_prediction(fname)
        assert (emupredload.mean() == emupred.mean()).all()

        os.remove(fname)
