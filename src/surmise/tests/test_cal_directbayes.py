import numpy as np
import pytest
from surmise.emulation import emulator
from surmise.calibration import calibrator
from .conftest import does_not_raise
from .shared_scenario import y_td as y, obsvar_td as obsvar, \
    x_std, theta_ball as theta, x_range, theta_range, prior_balldrop, timedrop, RNG_SEED
# TODO: TEMPORARY FIX
import surmise
surmise.set_RNG(np.random.default_rng(RNG_SEED))

# Use set RNG in this entire test module
pytestmark = pytest.mark.usefixtures('seeded_rng')

METHOD_IN_TEST = 'directbayes'

##############################################
#            Simple scenarios                #
##############################################
# Obtain computer model output via filtered data
f = timedrop(x_std, theta, x_range, theta_range)

# Fit an emulator via non-filtered data
emulator_nf_1 = emulator(x=x_std, theta=theta, f=f, method='PCGP')
pred_nf = emulator_nf_1.predict(x=x_std, theta=theta)
pred_nf_mean = pred_nf.mean()

# Filter out the data
ys = 1 - np.sum((pred_nf_mean - y) ** 2, 0) / np.sum((y - np.mean(y)) ** 2, 0)
theta_f = theta[ys > 0.5]

# Obtain computer model output via filtered data
f_f = timedrop(x_std, theta_f, x_range, theta_range)

# Fit an emulator via filtered data
emulator_f_1 = emulator(x=x_std, theta=theta_f, f=f_f, method='PCGP')
# emulator_f_2 = emulator(x=x_std, theta=theta_f, f=f_f, method='PCGP')

args2 = {'theta0': np.array([[0.4]]),
         'numsamp': 20,
         'stepType': 'normal',
         'stepParam': [0.4]}
args3 = {'theta0': np.array([[0.4]]),
         'stepParam': [0.4]}
args4 = {'theta0': np.array([[0.4]])}
args5 = {'stepParam': [0.4]}
args6 = {'sampler': 'metropolis_hastings'}


@pytest.mark.parametrize(
    "input2,expectation",
    [
        # (emulator_f_1, args1, does_not_raise()),
        # (emulator_f_2, args1, does_not_raise()),
        (args2, does_not_raise()),
        (args3, does_not_raise()),
        (args4, does_not_raise()),
        (args5, does_not_raise()),
        (args6, does_not_raise()),
    ],
)
def test_cal_MLcal(emu_timedrop, input2, expectation):
    with expectation:
        assert calibrator(emu=emu_timedrop,
                          y=y,
                          x=x_std,
                          thetaprior=prior_balldrop,
                          method=METHOD_IN_TEST,
                          yvar=obsvar,
                          args=input2) is not None


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
                          args=args2)
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
def test_call(emu_timedrop, expectation):
    cal = calibrator(emu=emu_timedrop,
                     y=y,
                     x=x_std,
                     thetaprior=prior_balldrop,
                     method=METHOD_IN_TEST,
                     yvar=obsvar,
                     args=args2)
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
                     args=args2)
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
                     args=args2)
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
                     args=args2)
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
                     args=args2)
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
                     args=args2)
    with expectation:
        assert cal(x_std) is not None
        assert repr(cal.theta()) is not None


def test_cal_noobsvar(emu_timedrop):
    with pytest.raises(ValueError):
        calibrator(emu=emu_timedrop,
                   y=y,
                   x=x_std,
                   thetaprior=prior_balldrop,
                   method=METHOD_IN_TEST,
                   # yvar=obsvar,
                   args=args2)
