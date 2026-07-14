import numpy as np
from surmise.emulation import emulator
from surmise.calibration import calibrator
import pytest

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin, f_lin, y_lin as y, \
                             obsvar_lin as obsvar, priorphys_lin, RNG_SEED

_rng = np.random.default_rng(RNG_SEED)
pytestmark = pytest.mark.usefixtures('seeded_rng')

METHOD_IN_TEST = 'directbayeswoodbury'

##############################################
#            Simple scenarios                #
##############################################

emulator_1 = emulator(x=x, theta=theta_lin, f=f_lin, method='PCGPwM')
emulator_w_grad = emulator(x=x, theta=theta_lin, f=f_lin,
                           method='PCGPwM',
                           args={'return_grad': True})


##############################################
# Unit tests to initialize an emulator class #
##############################################
# test to check none-type inputs
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     (emulator_w_grad, does_not_raise()),
     ],
    )
def test_cal_directbayes(input1, expectation):
    with expectation:
        assert calibrator(emu=input1,
                          y=y,
                          x=x,
                          thetaprior=priorphys_lin,
                          method=METHOD_IN_TEST,
                          yvar=obsvar) is not None


# test to check none-type inputs
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_predict(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.predict(x=x) is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_predict_mean(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    pred_bayes = cal_bayes.predict(x=x)
    with expectation:
        assert pred_bayes.mean() is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_predict_var(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    pred_bayes = cal_bayes.predict(x=x)
    with expectation:
        assert pred_bayes.var() is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_predict_rnd(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    pred_bayes = cal_bayes.predict(x=x)
    with expectation:
        assert pred_bayes.rnd() is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, pytest.raises(ValueError)),
     # (emulator_2, pytest.raises(ValueError)),
     ],
    )
def test_cal_predict_lpdf(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    pred_bayes = cal_bayes.predict(x=x)
    with expectation:
        assert pred_bayes.lpdf() is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_repr(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    thetadist_cal_bayes = cal_bayes.theta
    with expectation:
        assert repr(thetadist_cal_bayes) is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (None, does_not_raise()),
     (10, does_not_raise()),
     ],
    )
def test_cal_thetadist_call(input1, expectation):
    cal_bayes = calibrator(emu=emulator_1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta(s=input1) is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_mean(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta.mean() is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_var(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta.var() is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_rnd(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta.rnd() is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
     (emulator_1, does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_lpdf(input1, expectation):
    cal_bayes = calibrator(emu=input1,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta.lpdf(theta=theta_lin) is not None
