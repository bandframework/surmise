# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
from boreholetestfunctions import borehole_failmodel, borehole_true
from surmise.emulation import emulator
from surmise.calibration import calibrator


class thetaprior:
    """ This defines the class instance of priors provided to the methods. """
    def lpdf(theta):
        return (np.sum(sps.norm.logpdf(theta, 1, 0.5), 1)).reshape((len(theta), 1))

    def rnd(n):
        return np.vstack((sps.norm.rvs(1, 0.5, size=(n, 4))))


x = sps.uniform.rvs(0, 1, [50, 3])
x[:, 2] = x[:, 2] > 0.5
yt = np.squeeze(borehole_true(x))
yvar = (10 ** (-2)) * np.ones(yt.shape)
thetatot = (thetaprior.rnd(15))
f = (borehole_failmodel(x, thetatot).T).T
y = yt + sps.norm.rvs(0, np.sqrt(yvar))
emu = emulator(x, thetatot, f, method='PCGPwM', options={'xrmnan': 'all',
                                                         'thetarmnan': 'never',
                                                         'return_grad': True})

emu.fit()
cal = calibrator(emu, y, x, thetaprior, yvar, method='directbayes')
print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis=0), 3))

pending = np.full(f.shape, False)
thetachoices = cal.theta(200)
numnewtheta = 10
choicescost = np.ones(thetachoices.shape[0])
thetaneworig, info = emu.supplement(size=numnewtheta, thetachoices=thetachoices,
                                    choicescost=choicescost,
                                    cal=cal, overwrite=True,
                                    args={'includepending': True,
                                          'costpending': 0.01+0.99*np.mean(pending, 0),
                                          'pending': pending})
thetaneworig = thetaneworig[:numnewtheta, :]
thetanew = thetaneworig


##############################################
# Unit tests to initialize an emulator class #
##############################################


@contextmanager
def does_not_raise():
    yield


@pytest.fixture
def cmdopt1(request):
    return request.config.getoption("--cmdopt1")


# tests for prediction class methods:
# test to check the prediction.mean()
def test_prediction_mean(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    try:
        pred.mean()
    except:
        pytest.fail('mean() functionality does not exist in the method')

# test to check the prediction.var()
def test_prediction_var(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    try:
        pred.var()
    except:
        pytest.fail('var() functionality does not exist in the method')

# test to check the prediction.covx()
def test_prediction_covx(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    try:
        pred.covx()
    except:
        pytest.fail('covx() functionality does not exist in the method')

# test to check the prediction.covxhalf()
def test_prediction_covxhalf(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    try:
        pred.covxhalf()
    except:
        pytest.fail('covxhalf() functionality does not exist in the method')

# test to check the prediction.mean_gradtheta()
def test_prediction_mean_gradtheta(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    try:
        pred.mean_gradtheta()
    except:
        pytest.fail('mean_gradtheta() functionality does not exist in'
                    ' the method')

# test to check the prediction.covx_gradtheta()
def test_prediction_covxhalf_gradtheta(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    try:
        pred.covxhalf_gradtheta()
    except:
        pytest.fail('covxhalf_gradtheta() functionality does not exist in'
                    ' the method')

# test to check the prediction.rnd()
def test_prediction_rnd(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    try:
        pred.rnd()
    except:
        pytest.fail('rnd() functionality does not exist in the method')

# test to check the prediction.lpdf()
def test_prediction_lpdf(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    pred = emu.predict(x=x, theta=theta)
    try:
        pred.lpdf()
    except:
        pytest.fail('lpdf() functionality does not exist in the method')

# test to check emulator.remove()
def test_remove(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    emu.remove(theta=theta1)
    assert len(emu._emulator__theta) == 25, 'Check emulator.remove()'

# test to check emulator.remove() with a calibrator
# def test_remove_cal(cmdopt1):
#    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
#    cal_bayes = calibrator(emu=emu,
#                           y=y,
#                           x=x,
#                           thetaprior=priorphys_lin,
#                           method='directbayes',
#                           yvar=obsvar)
#    emu.remove(theta=theta1, cal=cal_bayes)
#    assert len(emu._emulator__theta) <= 50, 'Check emulator.remove() with'
#    ' calibration'

# test to check emulator.update()
def test_update(cmdopt1):
    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    thetanew = priorphys_lin.rnd(10)
    fnew = balldropmodel_linear(xv, thetanew)
    emu.update(x=None, theta=thetanew, f=fnew)
    assert len(emu._emulator__theta) == 60, 'Check emulator.update()'

# test to check emulator.supplement()
#def test_supplement(cmdopt1):
#    emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
#    thetanew = priorphys_lin.rnd(10)
#    fnew = balldropmodel_linear(xv, thetanew)
#    emu.update(x=None, theta=thetanew, f=fnew)
#    assert len(emu._emulator__theta) == 60, 'Check emulator.update()'
