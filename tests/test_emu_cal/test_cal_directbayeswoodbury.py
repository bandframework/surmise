import numpy as np
import scipy.stats as sps
import pytest
from contextlib import contextmanager
from surmise.emulation import emulator
from surmise.calibration import calibrator

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
    def lpdf(theta):
        return (sps.norm.logpdf(theta[:, 0], 0, 5) +
                              sps.gamma.logpdf(theta[:, 1], 2, 0, 10)).reshape((len(theta), 1))

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n),
                          sps.gamma.rvs(2, 0, 10, size=n))).T


theta_lin = priorphys_lin.rnd(50)
f_lin = balldropmodel_linear(xv, theta_lin)


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
emulator_1 = emulator(x=x, theta=theta_lin, f=f_lin, method='PCGP')
#emulator_2 = emulator(x=x, theta=theta_lin, f=f_lin, method='PCGP')

##############################################
# Unit tests to initialize an emulator class #
##############################################


@contextmanager
def does_not_raise():
    yield


# # test to check none-type inputs
# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_directbayes(input1, expectation):
#     with expectation:
#         assert calibrator(emu=input1,
#                           y=y,
#                           x=x,
#                           thetaprior=priorphys_lin,
#                           method='directbayeswoodbury',
#                           yvar=obsvar) is not None


# # test to check none-type inputs
# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_predict(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     with expectation:
#         assert cal_bayes.predict(x=x) is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_predict_mean(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     pred_bayes = cal_bayes.predict(x=x)
#     with expectation:
#         assert pred_bayes.mean() is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_predict_var(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     pred_bayes = cal_bayes.predict(x=x)
#     with expectation:
#         assert pred_bayes.var() is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_predict_rnd(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     pred_bayes = cal_bayes.predict(x=x)
#     with expectation:
#         assert pred_bayes.rnd() is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, pytest.raises(ValueError)),
#      # (emulator_2, pytest.raises(ValueError)),
#      ],
#     )
# def test_cal_predict_lpdf(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     pred_bayes = cal_bayes.predict(x=x)
#     with expectation:
#         assert pred_bayes.lpdf() is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_thetadist(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     with expectation:
#         assert cal_bayes.theta is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_thetadist_repr(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     thetadist_cal_bayes = cal_bayes.theta
#     with expectation:
#         assert repr(thetadist_cal_bayes) is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (None, does_not_raise()),
#      (10, does_not_raise()),
#      ],
#     )
# def test_cal_thetadist_call(input1, expectation):
#     cal_bayes = calibrator(emu=emulator_1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     with expectation:
#         assert cal_bayes.theta(s=input1) is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_thetadist_mean(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     with expectation:
#         assert cal_bayes.theta.mean() is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_thetadist_var(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     with expectation:
#         assert cal_bayes.theta.var() is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_thetadist_rnd(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     with expectation:
#         assert cal_bayes.theta.rnd() is not None


# @pytest.mark.parametrize(
#     "input1,expectation",
#     [
#      (emulator_1, does_not_raise()),
#      # (emulator_2, does_not_raise()),
#      ],
#     )
# def test_cal_thetadist_lpdf(input1, expectation):
#     cal_bayes = calibrator(emu=input1,
#                            y=y,
#                            x=x,
#                            thetaprior=priorphys_lin,
#                            method='directbayeswoodbury',
#                            yvar=obsvar)
#     with expectation:
#         assert cal_bayes.theta.lpdf(theta=theta_lin) is not None
