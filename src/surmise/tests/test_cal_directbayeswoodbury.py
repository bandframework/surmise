from surmise.calibration import calibrator
import pytest

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin, y_lin as y, \
                             obsvar_lin as obsvar, priorphys_lin

pytestmark = pytest.mark.usefixtures('seeded_rng', '_session_rng')

METHOD_IN_TEST = 'directbayeswoodbury'


##############################################
#            Simple scenarios                #
##############################################
# test to check none-type inputs
@pytest.mark.parametrize(
    "grad_flag,expectation",
    [
     (False, does_not_raise()),
     (True, does_not_raise()),
     ],
    )
def test_cal_directbayes(grad_flag, expectation,
                         emu_lin_pcgpwm, emu_lin_pcgpwm_wgrad):
    emu = emu_lin_pcgpwm_wgrad if grad_flag else emu_lin_pcgpwm
    with expectation:
        assert calibrator(emu=emu,
                          y=y,
                          x=x,
                          thetaprior=priorphys_lin,
                          method=METHOD_IN_TEST,
                          yvar=obsvar) is not None


# test to check none-type inputs
@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_predict(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.predict(x=x) is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_predict_mean(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    pred_bayes = cal_bayes.predict(x=x)
    with expectation:
        assert pred_bayes.mean() is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_predict_var(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    pred_bayes = cal_bayes.predict(x=x)
    with expectation:
        assert pred_bayes.var() is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_predict_rnd(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    pred_bayes = cal_bayes.predict(x=x)
    with expectation:
        assert pred_bayes.rnd() is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (pytest.raises(ValueError)),
     # (emulator_2, pytest.raises(ValueError)),
     ],
    )
def test_cal_predict_lpdf(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    pred_bayes = cal_bayes.predict(x=x)
    with expectation:
        assert pred_bayes.lpdf() is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_repr(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
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
def test_cal_thetadist_call(input1, expectation, emu_lin_pcgpwm):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta(s=input1) is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_mean(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta.mean() is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_var(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta.var() is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_rnd(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta.rnd() is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     # (emulator_2, does_not_raise()),
     ],
    )
def test_cal_thetadist_lpdf(emu_lin_pcgpwm, expectation):
    cal_bayes = calibrator(emu=emu_lin_pcgpwm,
                           y=y,
                           x=x,
                           thetaprior=priorphys_lin,
                           method=METHOD_IN_TEST,
                           yvar=obsvar)
    with expectation:
        assert cal_bayes.theta.lpdf(theta=theta_lin) is not None
