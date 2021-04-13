import numpy as np
import scipy.stats as sps
from boreholetestfunctions import borehole_model, borehole_failmodel, borehole_true
from surmise.emulation import emulator
from surmise.calibration import calibrator
from time import time


def alg(thetaprior, n=25, maxthetas=500, flag_failmodel=True, obviate=True):
    if flag_failmodel is False:
        bh_model = borehole_model
    else:
        bh_model = borehole_failmodel

    # generate input and data
    np.random.seed(0)
    x = sps.uniform.rvs(0, 1, (n, 3))
    x[:, 2] = x[:, 2] > 0.5
    yt = np.squeeze(borehole_true(x))
    yvar = (10 ** (-2)) * np.ones(yt.shape)
    y = yt + sps.norm.rvs(0, np.sqrt(yvar))
    np.random.seed()

    thetatot = (thetaprior.rnd(15))
    f = bh_model(x, thetatot)
    # fit an emulator
    emu = emulator(x, thetatot, f, method='PCGPwM',
                   options={'xrmnan': 'all',
                            'thetarmnan': 'never',
                            'return_grad': True})

    # apply emulator to calibration
    cal = calibrator(emu, y, x, thetaprior, yvar, method='directbayeswoodbury')
    print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis=0), 3))

    thetaidqueue = np.zeros(0)
    xidqueue = np.zeros(0)
    pending = np.full(f.shape, False)
    complete = np.full(f.shape, True)
    cancelled = np.full(f.shape, False)
    numperbatch = 250

    numcompletetheta = thetatot.shape[0]
    numcomplete = []
    numpending = []
    numcancel = []
    looptime = []
    emutime = []
    caltime = []
    thetaquantile = []

    for k in range(0, 50):
        print('Percentage Cancelled: %0.2f ( %d / %d)' %
              (100*np.round(np.mean(1-pending-complete), 4),
               np.sum(1-pending-complete),
               np.prod(pending.shape)))
        print('Percentage Pending: %0.2f ( %d / %d)' %
              (100*np.round(np.mean(pending), 4),
               np.sum(pending),
               np.prod(pending.shape)))
        print('Percentage Complete: %0.2f ( %d / %d)' %
              (100*np.round(np.mean(complete), 4),
               np.sum(complete),
               np.prod(pending.shape)))

        numcomplete.append(complete.sum())
        numpending.append(pending.sum())
        numcancel.append(cancelled.sum())
        looptime.append(time())

        numnewtheta = 10
        keepadding = True
        while keepadding and (k > -1):
            print('f shape:', f.shape)
            print('pend shape:', pending.shape)
            numnewtheta += 2
            thetachoices = cal.theta(200)
            choicescost = np.ones(thetachoices.shape[0])
            thetaneworig, info = emu.supplement(size=numnewtheta,
                                                thetachoices=thetachoices,
                                                choicescost=choicescost,
                                                cal=cal,
                                                overwrite=True,
                                                args={'includepending': True,
                                                      'costpending': 0.01+0.99*np.mean(pending, 0),
                                                      'pending': pending})
            thetaneworig = thetaneworig[:numnewtheta, :]
            thetanew = thetaneworig

            # obviate if suggested
            if obviate:
                if info['obviatesugg'].shape[0] > 0:
                    pending[:, info['obviatesugg']] = False
                    print('obviating')
                    print(info['obviatesugg'])
                    for k in info['obviatesugg']:
                        queue2delete = np.where(thetaidqueue == k)[0]
                        if queue2delete.shape[0] > 0.5:
                            thetaidqueue = np.delete(thetaidqueue, queue2delete, 0)
                            xidqueue = np.delete(xidqueue, queue2delete, 0)
                            numcancel[-1] += 1

            if (thetanew.shape[0] > 0.5) and \
                (np.sum(np.hstack((pending,np.full((x.shape[0],thetanew.shape[0]),True)))) > 600):
                pending = np.hstack((pending,np.full((x.shape[0],thetanew.shape[0]),True)))
                complete = np.hstack((complete,np.full((x.shape[0],thetanew.shape[0]),False)))
                f = np.hstack((f,np.full((x.shape[0],thetanew.shape[0]),np.nan)))
                thetaidnewqueue = np.tile(np.arange(thetatot.shape[0], thetatot.shape[0]+
                                                    thetanew.shape[0]), (x.shape[0]))
                thetatot = np.vstack((thetatot,thetanew))
                xidnewqueue = np.repeat(np.arange(0,x.shape[0]), thetanew.shape[0], axis =0)
                if thetaidqueue.shape[0] == 0:
                    thetaidqueue = thetaidnewqueue
                    xidqueue = xidnewqueue
                else:
                    thetaidqueue = np.append(thetaidqueue, thetaidnewqueue)
                    xidqueue = np.append(xidqueue, xidnewqueue)
                keepadding = False

        priorityscore = np.zeros(thetaidqueue.shape)
        priorityscore = np.random.choice(np.arange(0, priorityscore.shape[0]),
                                         size=priorityscore.shape[0], replace=False)
        queuerearr = np.argsort(priorityscore)
        xidqueue = xidqueue[queuerearr]
        thetaidqueue = thetaidqueue[queuerearr]

        for l in range(0, np.minimum(xidqueue.shape[0], numperbatch)):
            f[xidqueue[l], thetaidqueue[l]] = bh_model(x[xidqueue[l], :],
                                                                 thetatot[thetaidqueue[l], :])
            pending[xidqueue[l], thetaidqueue[l]] = False
            complete[xidqueue[l], thetaidqueue[l]] = True
        print('sum of variance:')
        print(np.nansum(np.nanvar(f, 1)))
        thetaidqueue = np.delete(thetaidqueue, range(0, numperbatch), 0)
        xidqueue = np.delete(xidqueue, range(0, numperbatch), 0)

        numcompletetheta += numperbatch / n
        print('number of completed theta:', numcompletetheta)

        emustart = time()
        emu.update(theta=thetatot, f=f)
        emuend = time()
        cal.fit()
        calend = time()
        emutime.append(emuend - emustart)
        caltime.append(calend - emuend)

        thetarng = np.quantile(cal.theta.rnd(10000), (0.025, 0.5, 0.975), axis=0)
        thetaquantile.append(thetarng)

        print(np.round(thetarng, 3))

        # maximum parameter budget
        if numcompletetheta > maxthetas:
            print('exit with f shape: ', f.shape)
            break
    return cal, emu, {'ncomp': numcomplete, 'npend': numpending, 'ncancel': numcancel, 'looptime': looptime, 'emutime': emutime, 'caltime': caltime, 'quantile': thetaquantile}

#%% prior class
class thetaprior:
    """Prior class."""

    def lpdf(theta):
        """Return log density."""
        return (np.sum(sps.beta.logpdf(theta, 2, 2), 1)).reshape((len(theta), 1))

    def rnd(n):
        """Return random variables from the density."""
        return np.vstack((sps.beta.rvs(2, 2, size=(n, 4))))


# %% true posterior
n = 25

# generate input and data
np.random.seed(0)
x = sps.uniform.rvs(0, 1, (n, 3))
x[:, 2] = x[:, 2] > 0.5
yt = np.squeeze(borehole_true(x))
yvar = (10 ** (-2)) * np.ones(yt.shape)
y = yt + sps.norm.rvs(0, np.sqrt(yvar))
np.random.seed()

# fit an emulator
pass_emu = emulator(x, passthroughfunc=borehole_model, method='PCGPwM',
                    options={'xrmnan': 'all',
                             'thetarmnan': 'never',
                             'return_grad': True})

# apply emulator to calibration
true_cal = calibrator(pass_emu, y, x, thetaprior, yvar, method='directbayeswoodbury')
postthetas = true_cal.theta.rnd(10000)
postthetarng = np.quantile(postthetas, (0.025, 0.5, 0.975), axis=0)

# %% runs without failures, without obviation
cal, emu, res = alg(thetaprior, maxthetas=200, flag_failmodel=False, obviate=False)
sampthetas = cal.theta.rnd(10000)
sampthetarng = np.quantile(postthetas, (0.025, 0.5, 0.975), axis=0)

#%% quantile comparisons
print('estimated quantile from the last loop:\n', np.round(res['quantile'][-1][(0,-1),:], 3))
print('estimated posterior quantile:\n', np.round(sampthetarng[(0,-1),:], 3))
print('true posterior quantile:\n', np.round(postthetarng[(0,-1), :], 3))

print('\n', np.round(postthetarng[(0,-1), :] - sampthetarng[(0,-1),:], 3))
