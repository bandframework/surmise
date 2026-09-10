import numpy as np
import pytest
from surmise.calibration import calibrator
from .conftest import does_not_raise
from .shared_scenario import y_td as y, obsvar_td as obsvar, \
    x_std, theta_ball as theta, x_range, theta_range, prior_balldrop, timedrop, DEFAULT_MH_SPECS

# Use set RNG in this entire test module
pytestmark = pytest.mark.usefixtures('seeded_rng', '_session_rng')

METHOD_IN_TEST = 'directbayes'

##############################################
#            Simple scenarios                #
##############################################
# Obtain computer model output via filtered data
f = timedrop(x_std, theta, x_range, theta_range)

args_w_missing_specs = {'theta0': np.array([[0.4]]),
                        'numsamp': 20,
                        'stepType': 'normal',
                        'stepParam': [0.4]}


@pytest.mark.parametrize(
    "args,expectation",
    [
        (args_w_missing_specs, pytest.raises(ValueError)),
        (DEFAULT_MH_SPECS, does_not_raise()),
    ],
)
def test_cal_MLcal(emu_timedrop, args, expectation):
    with expectation:
        assert calibrator(emu=emu_timedrop,
                          y=y,
                          x=x_std,
                          thetaprior=prior_balldrop,
                          method=METHOD_IN_TEST,
                          yvar=obsvar,
                          args=args) is not None


@pytest.mark.parametrize(
    "input1,expectation",
    [
        (x_std, does_not_raise()),
        (None, does_not_raise()),
    ],
)
def test_cal_predict(emu_timedrop, input1, expectation):
    cal_test = calibrator(emu=emu_timedrop,
                          y=y,
                          x=x_std,
                          thetaprior=prior_balldrop,
                          method=METHOD_IN_TEST,
                          yvar=obsvar,
                          args=DEFAULT_MH_SPECS)
    with expectation:
        assert cal_test.predict(x=input1) is not None


@pytest.mark.parametrize(
    "expectation",
    [
        (does_not_raise()),
    ],
)
def test_repr(emu_timedrop, expectation):
    cal = calibrator(emu=emu_timedrop,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method=METHOD_IN_TEST,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS)
    pred_test = cal.predict(x=x_std)
    with expectation:
        assert repr(pred_test) is not None


@pytest.mark.parametrize(
    "expectation",
    [
        (does_not_raise()),
    ],
)
def test_call(emu_timedrop, expectation):
    cal = calibrator(emu=emu_timedrop,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method=METHOD_IN_TEST,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS)
    pred_test = cal.predict(x=x_std)
    with expectation:
        assert pred_test() is not None


@pytest.mark.parametrize(
    "expectation",
    [
        (does_not_raise()),
    ],
)
def test_meanvar(emu_timedrop, expectation):
    cal = calibrator(emu=emu_timedrop,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method=METHOD_IN_TEST,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS)
    pred_test = cal.predict(x=x_std)
    with expectation:
        assert pred_test.mean() is not None
        assert pred_test.var() is not None


@pytest.mark.parametrize(
    "expectation",
    [
        (does_not_raise()),
    ],
)
def test_thetalpdf(emu_timedrop, expectation):
    cal = calibrator(emu=emu_timedrop,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method=METHOD_IN_TEST,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS)
    logpost = cal.theta.lpdf(theta=theta)
    with expectation:
        assert logpost is not None


@pytest.mark.parametrize(
    "expectation",
    [
        (does_not_raise()),
    ],
)
def test_pred(emu_timedrop, expectation):
    cal = calibrator(emu=emu_timedrop,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method=METHOD_IN_TEST,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS)
    pred_test = cal.predict(x=x_std)
    with expectation:
        assert pred_test.rnd(10) is not None
        assert pred_test(10) is not None


@pytest.mark.parametrize(
    "expectation",
    [
        (does_not_raise()),
    ],
)
def test_theta_meanvar(emu_timedrop, expectation):
    cal = calibrator(emu=emu_timedrop,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method=METHOD_IN_TEST,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS)
    with expectation:
        assert cal.theta.mean(args=None) is not None
        assert cal.theta.var(args=None) is not None
        assert cal.theta.rnd(10) is not None


@pytest.mark.parametrize(
    "expectation",
    [
        (does_not_raise()),
    ],
)
def test_cal_repr(emu_timedrop, expectation):
    cal = calibrator(emu=emu_timedrop,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method=METHOD_IN_TEST,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS)
    with expectation:
        assert cal(x_std) is not None
        assert repr(cal.theta()) is not None
