import numpy as np
import pytest
from surmise.emulation import emulator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f, \
    theta_new_lin as thetanew, f_new_lin as fnew

pytestmark = pytest.mark.usefixtures('seeded_rng')

##############################################
#            Simple scenarios                #
##############################################
f1 = f[0:15, :]
f2 = f[:, 0:25]
theta1 = theta[0:25, :]
x1 = x[0:15, :]
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)
simsd = 1e-3 * np.ones_like(f)


# tests for prediction class methods:
# test to check the prediction.mean()
@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
def test_prediction_mean(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.mean() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.var()
def test_prediction_var(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.var() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.covx()
def test_prediction_covx(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.covx() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.covxhalf()
def test_prediction_covxhalf(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    with expectation:
        assert pred.covxhalf() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', pytest.raises(ValueError)),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.mean_gradtheta()
def test_prediction_mean_gradtheta(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta, args={'return_grad': True})
    with expectation:
        assert pred.mean_gradtheta() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', pytest.raises(ValueError)),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# test to check the prediction.covx_gradtheta()
def test_prediction_covxhalf_gradtheta(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta, args={'return_grad': True})
    with expectation:
        assert pred.covxhalf_gradtheta() is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', pytest.raises(ValueError))  # PCSK does not support remove function
    ],
    )
# test to check emulator.remove()
def test_remove(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)

    with expectation:
        emu.remove(theta=theta1)
        assert len(emu._emulator__theta) == 25, 'Check emulator.remove()'


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', pytest.raises(IndexError))  # PCSK does not support update function
    ],
    )
# test to check emulator.update()
def test_update(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    with expectation:
        emu.update(x=None, theta=thetanew, f=fnew)
        assert len(emu._emulator__theta) == 60, 'Check emulator.update()'
