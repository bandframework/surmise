import pytest
from surmise.emulation import emulator
from surmise.calibration import calibrator
from .conftest import does_not_raise
from .shared_scenario import x_td as x, y_td as y, obsvar_td as obsvar, \
    x_std, theta_ball, x_range, theta_range, prior_balldrop, timedrop, \
    DEFAULT_MH_SPECS

pytestmark = pytest.mark.usefixtures('seeded_rng')

##############################################
#            Simple scenarios                #
##############################################
# Obtain computer model output via filtered data
f = timedrop(x_std, theta_ball, x_range, theta_range)


@pytest.mark.parametrize(
    "cmdopt2,expectation",
    [
     ('directbayes', does_not_raise()),
     ('directbayeswoodbury', does_not_raise()),
    ],
    )
# tests for prediction class methods:
# test to check the prediction.mean()
def test_prediction_mean(cmdopt2, expectation):
    emu = emulator(x=x, theta=theta_ball, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS)
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
    emu = emulator(x=x, theta=theta_ball, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS
                     )
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
    emu = emulator(x=x, theta=theta_ball, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS
                     )
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
    emu = emulator(x=x, theta=theta_ball, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS
                     )
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
    emu = emulator(x=x, theta=theta_ball, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS
                     )
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
    emu = emulator(x=x, theta=theta_ball, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS
                     )
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
    emu = emulator(x=x, theta=theta_ball, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS
                     )
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
    emu = emulator(x=x, theta=theta_ball, f=f, method='PCGPwM')
    cal = calibrator(emu=emu,
                     y=y,
                     x=x,
                     thetaprior=prior_balldrop,
                     method=cmdopt2,
                     yvar=obsvar,
                     args=DEFAULT_MH_SPECS
                     )
    with expectation:
        cal.theta.lpdf(theta=theta_ball)
