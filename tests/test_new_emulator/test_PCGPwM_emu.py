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
