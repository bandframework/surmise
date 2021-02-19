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
        h0 = 50 + theta[k, 0]
        vter = theta[k, 1]
        f[k, :] = h0 - vter * t
    return f.T


tvec = np.concatenate((np.arange(0.1, 4.3, 0.1), np.arange(0.1, 4.3, 0.1)))
x = np.array([[0.1],
              [0.2],
              [0.3],
              [0.4],
              [0.5],
              [0.6],
              [0.7],
              [0.8],
              [0.9],
              [1.0],
              [1.2],
              [2.6],
              [2.9],
              [3.1],
              [3.3],
              [3.5],
              [3.7], ]).astype('object')
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
x = x.reshape(17, )
##############################################
# Unit tests to initialize an emulator class #
##############################################


@contextmanager
def does_not_raise():
    yield


# test to check the predict with 1d-x
@pytest.mark.parametrize(
    "input1,input2,expectation",
    [
     (x, theta, does_not_raise()),
     (x.reshape((17, 1)), theta, pytest.raises(ValueError)),
     ],
    )
def test_predict_multi(input1, input2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu.predict(x=input1, theta=input2) is not None
