import numpy as np
import scipy.stats as sps
import pytest
from contextlib import contextmanager
from surmise.emulation import emulator
from surmise.calibration import calibrator

##############################################
#            Simple scenarios                #
##############################################

# height
x = np.array([[0.178, 0.356, 0.534, 0.712, 0.89, 1.068, 1.246, 1.424, 1.602,
               1.78, 1.958, 2.67, 2.848, 3.026, 3.204, 3.382, 3.56, 3.738,
               3.916, 4.094, 4.272]]).T

# time
y = np.array([[0.27, 0.22, 0.27, 0.43, 0.41, 0.49, 0.46, 0.6, 0.65, 0.62, 0.7,
               0.81, 0.69, 0.81, 0.89, 0.86, 0.89, 1.1, 1.05, 0.99, 1.05]]).T
obsvar = np.maximum(0.2 * y, 0.1)


# Computer implementation of the mathematical model
def timedrop(x, theta, hr, gr):
    # Assume x and theta are within (0, 1)
    min_g = min(gr)
    range_g = max(gr) - min(gr)
    min_h = min(hr)
    range_h = max(hr) - min_h
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        g = range_g * theta[k] + min_g
        h = range_h * x + min_h
        f[k, :] = np.sqrt(2 * h / g).reshape(x.shape[0])
    return f.T


# Define prior
class prior_balldrop:
    """ This defines the class instance of priors provided to the method. """

    def lpdf(theta):
        return (sps.uniform.logpdf(theta[:, 0], 0, 1)).reshape((len(theta), 1))

    def rnd(n):
        return np.vstack((sps.uniform.rvs(0, 1, size=n)))


# Draw 100 random parameters from uniform prior
n = 100
theta = prior_balldrop.rnd(n).reshape(n, 1)
theta_range = np.array([1, 30])

# Standardize
x_range = np.array([min(x), max(x)])
x_std = (x - min(x)) / (max(x) - min(x))

# Obtain computer model output via filtered data
f = timedrop(x_std, theta, x_range, theta_range)


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "cmdopt2,expectation",
    [
     ('directbayes', does_not_raise()),
     ('directbayeswoodbury', does_not_raise()),
     # ('mlbayeswoodbury', does_not_raise())
    ],
    )
# tests for prediction class methods:
# test to check the prediction.mean()
def test_prediction_mean(cmdopt2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar)
    pred = cal.predict(x=x)
    with expectation:
        pred.mean()


@pytest.mark.parametrize(
    "cmdopt2,expectation",
    [
     ('directbayes', does_not_raise()),
     ('directbayeswoodbury', does_not_raise()),
     # ('mlbayeswoodbury', does_not_raise())
    ],
    )
# test to check the prediction.var()
def test_prediction_var(cmdopt2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar)
    pred = cal.predict(x=x)
    with expectation:
        pred.var()


@pytest.mark.parametrize(
    "cmdopt2,expectation",
    [
     ('directbayes', does_not_raise()),
     ('directbayeswoodbury', does_not_raise()),
     # ('mlbayeswoodbury', does_not_raise())
    ],
    )
# test to check the prediction.rnd()
def test_prediction_rnd(cmdopt2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar)
    pred = cal.predict(x=x)
    with expectation:
        pred.rnd()


@pytest.mark.parametrize(
    "cmdopt2,expectation",
    [
     ('directbayes', pytest.raises(ValueError)),
     ('directbayeswoodbury', pytest.raises(ValueError)),
     # ('mlbayeswoodbury', pytest.raises(ValueError))
    ],
    )
# test to check the prediction.lpdf()
def test_prediction_lpdf(cmdopt2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar)
    pred = cal.predict(x=x)
    with expectation:
        pred.lpdf()


@pytest.mark.parametrize(
    "cmdopt2,expectation",
    [
     ('directbayes', does_not_raise()),
     ('directbayeswoodbury', does_not_raise()),
     # ('mlbayeswoodbury', does_not_raise())
    ],
    )
# test to check the theta.mean()
def test_prediction_thetamean(cmdopt2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar)
    with expectation:
        cal.theta.mean()


@pytest.mark.parametrize(
    "cmdopt2,expectation",
    [
     ('directbayes', does_not_raise()),
     ('directbayeswoodbury', does_not_raise()),
     # ('mlbayeswoodbury', does_not_raise())
    ],
    )
# test to check the theta.var()
def test_prediction_thetavar(cmdopt2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar)
    with expectation:
        cal.theta.var()


@pytest.mark.parametrize(
    "cmdopt2,expectation",
    [
     ('directbayes', does_not_raise()),
     ('directbayeswoodbury', does_not_raise()),
     # ('mlbayeswoodbury', does_not_raise())
    ],
    )
# test to check the theta.rnd()
def test_prediction_thetarnd(cmdopt2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar)
    with expectation:
        cal.theta.rnd()


@pytest.mark.parametrize(
    "cmdopt2,expectation",
    [
     ('directbayes', does_not_raise()),
     ('directbayeswoodbury', does_not_raise()),
     # ('mlbayeswoodbury', does_not_raise())
    ],
    )
# test to check the theta.lpdf()
def test_prediction_thetalpdf(cmdopt2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar)
    with expectation:
        cal.theta.lpdf(theta=theta)
