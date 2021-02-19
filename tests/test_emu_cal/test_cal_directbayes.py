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
obsvar = np.maximum(0.2*y, 0.1)


def timedrop(x, theta, hr, gr):
    '''Computer implementation of the mathematical model'''
    # Assume x and theta are within (0, 1)
    min_g = min(gr)
    range_g = max(gr) - min(gr)
    min_h = min(hr)
    range_h = max(hr) - min_h
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        g = range_g*theta[k] + min_g
        h = range_h*x + min_h
        f[k, :] = np.sqrt(2*h/g).reshape(x.shape[0])
    return f.T


class prior_balldrop:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        return sps.uniform.logpdf(theta[:, 0], 0, 1).reshape((len(theta), 1))

    def rnd(n):
        return np.vstack((sps.uniform.rvs(0, 1, size=n)))


# Draw 100 random parameters from uniform prior
n = 100
theta = prior_balldrop.rnd(n)
theta_range = np.array([1, 30])

# Standardize
x_range = np.array([min(x), max(x)])
x_std = (x - min(x))/(max(x) - min(x))

# Obtain computer model output via filtered data
f = timedrop(x_std, theta, x_range, theta_range)

# Fit an emulator via non-filtered data
emulator_nf_1 = emulator(x=x_std, theta=theta, f=f, method='PCGP')
pred_nf = emulator_nf_1.predict(x=x_std, theta=theta)
pred_nf_mean = pred_nf.mean()

# Filter out the data
ys = 1 - np.sum((pred_nf_mean - y)**2, 0)/np.sum((y - np.mean(y))**2, 0)
theta_f = theta[ys > 0.5]

# Obtain computer model output via filtered data
f_f = timedrop(x_std, theta_f, x_range, theta_range)

# Fit an emulator via filtered data
emulator_f_1 = emulator(x=x_std, theta=theta_f, f=f_f, method='PCGP')
#emulator_f_2 = emulator(x=x_std, theta=theta_f, f=f_f, method='PCGP')

args2 = {'theta0': np.array([[0.4]]),
         'numsamp': 20,
         'stepType': 'normal',
         'stepParam': [0.4]}
args3 = {'theta0': np.array([[0.4]]),
         'stepParam': [0.4]}
args4 = {'theta0': np.array([[0.4]])}
args5 = {'stepParam': [0.4]}


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "input1,input2,expectation",
    [
     # (emulator_f_1, args1, does_not_raise()),
     # (emulator_f_2, args1, does_not_raise()),
     (emulator_f_1, args2, does_not_raise()),
     (emulator_f_1, args3, does_not_raise()),
     (emulator_f_1, args4, does_not_raise()),
     (emulator_f_1, args5, does_not_raise()),
     ],
    )
def test_cal_MLcal(input1, input2, expectation):
    with expectation:
        assert calibrator(emu=input1,
                          y=y,
                          x=x_std,
                          thetaprior=prior_balldrop,
                          method='directbayes',
                          yvar=obsvar,
                          args=input2) is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (x_std, does_not_raise()),
     (None, does_not_raise()),
     ],
    )
def test_cal_predict(input1, expectation):
    cal_test = calibrator(emu=emulator_f_1,
                          y=y,
                          x=x_std,
                          thetaprior=prior_balldrop,
                          method='directbayes',
                          yvar=obsvar,
                          args=args2)
    with expectation:
        assert cal_test.predict(x=input1) is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_repr(expectation):
    cal = calibrator(emu=emulator_f_1,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method='directbayes',
                     yvar=obsvar,
                     args=args2)
    pred_test = cal.predict(x=x_std)
    with expectation:
        assert repr(pred_test) is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_call(expectation):
    cal = calibrator(emu=emulator_f_1,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method='directbayes',
                     yvar=obsvar,
                     args=args2)
    pred_test = cal.predict(x=x_std)
    with expectation:
        assert pred_test() is not None
