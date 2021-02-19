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


# Followings are the tests to check the input configurations
# test to check none-type inputs
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
     (x, theta, f, does_not_raise()),
     (x, None, f, pytest.raises(ValueError)),  # has not developed yet
     (None, theta, f, does_not_raise()),
     (x, theta, None, pytest.raises(ValueError)),
     (x, None, None, pytest.raises(ValueError)),
     (None, theta, None, pytest.raises(ValueError)),
     (None, None, f, pytest.raises(ValueError)),  # has not developed yet
     (None, None, None, pytest.raises(ValueError)),
     ],
    )
def test_none_input(input1, input2, input3, expectation):
    with expectation:
        assert emulator(x=input1,
                        theta=input2,
                        f=input3,
                        method='PCGP') is not None


# test to check the dimension of the inputs
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
     (x, theta, f, does_not_raise()),
     (x, theta, f.T, does_not_raise()),  # failure
     (x, None, f.T, pytest.raises(ValueError)),  # has not developed yet
     (x.T, theta, f, pytest.raises(ValueError)),
     (x.T, None, f, pytest.raises(ValueError)),
     (x, theta.T, f, pytest.raises(ValueError)),
     (x1, theta, f1, does_not_raise()),
     (x, theta, f1, pytest.raises(ValueError)),
     (x, theta, f2, pytest.raises(ValueError)),
     (x, theta1, f, pytest.raises(ValueError)),
     (None, theta1, f, pytest.raises(ValueError)),
     (None, theta, f.T, does_not_raise()),
     (x1, theta, f, pytest.raises(ValueError)),
     ],
    )
def test_size_input(input1, input2, input3, expectation):
    with expectation:
        assert emulator(x=input1,
                        theta=input2,
                        f=input3,
                        method='PCGP') is not None


# test to check the dimension of the inputs
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
     (x, theta, f0d, pytest.raises(ValueError)),
     (x0d, theta, f, pytest.raises(ValueError)),
     (x, theta0d, f, pytest.raises(ValueError)),
     ],
    )
def test_0d_input(input1, input2, input3, expectation):
    with expectation:
        assert emulator(x=input1,
                        theta=input2,
                        f=input3,
                        method='PCGP') is not None
# TO DO: Add tests for univariate data
# TO DO: Add tests for data including NAs and infs


# Following are the tests to check the emulator method configs
# test to check if an emulator module is imported
@pytest.mark.parametrize(
    "example_input,expectation",
    [
     ('PCGP', does_not_raise()),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_method1(example_input, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method=example_input) is not None


# test to check if 'thetareps' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (True, does_not_raise()),
     (False, does_not_raise()),
     (0, pytest.raises(ValueError)),
     (1, pytest.raises(ValueError)),
     (0.5, pytest.raises(ValueError)),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options1(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'thetareps': input1}) is not None


# test to check if 'xreps' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (True, does_not_raise()),
     (False, does_not_raise()),
     (0,  pytest.raises(ValueError)),
     (1,  pytest.raises(ValueError)),
     (0.5,  pytest.raises(ValueError)),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options2(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'xreps': input1}) is not None


# test to check if 'thetarmnan' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     #(True, does_not_raise()),
     (False, does_not_raise()),
     #(0, does_not_raise()),
     (1, does_not_raise()),
     (0.5, does_not_raise()),
     (2, pytest.raises(ValueError)),
     #('any', does_not_raise()),
     ('some', does_not_raise()),
     ('most', does_not_raise()),
     ('alot', does_not_raise()),
     ('all', does_not_raise()),
     ('never', does_not_raise()),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options3(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'thetarmnan': input1}) is not None


# test to check if 'xrmnan' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     #(True, does_not_raise()),
     (False, does_not_raise()),
     #(0, does_not_raise()),
     (1, does_not_raise()),
     (0.5, does_not_raise()),
     (2, pytest.raises(ValueError)),
     #('any', does_not_raise()),
     ('some', does_not_raise()),
     ('most', does_not_raise()),
     ('alot', does_not_raise()),
     ('all', does_not_raise()),
     ('never', does_not_raise()),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options4(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'xrmnan': input1}) is not None


# test to check if 'rmthetafirst' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (True, does_not_raise()),
     (False, does_not_raise()),
     (0, pytest.raises(ValueError)),
     (1, pytest.raises(ValueError)),
     (0.5, pytest.raises(ValueError)),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options5(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'rmthetafirst': input1}) is not None


# test to check if 'autofit' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (True, does_not_raise()),
     (False, does_not_raise()),
     (0, pytest.raises(ValueError)),
     (1, pytest.raises(ValueError)),
     (0.5, pytest.raises(ValueError)),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options6(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'autofit': input1}) is not None


# tests to check the emulator repr()
@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_repr(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert repr(emu) is not None


# tests to check the emulator call()
@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_call(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu(x=x, theta=theta) is not None


# tests to check the emulator args
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ({'epsilon': 1.5, 'hypregmean': -10, 'hypregLB': -20}, does_not_raise()),
     ],
    )
def test_args(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        args=input1) is not None
