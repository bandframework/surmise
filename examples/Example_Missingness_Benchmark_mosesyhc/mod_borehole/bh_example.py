import numpy as np
import scipy.stats as sps
from boreholetestfunctions import borehole_model, borehole_failmodel, borehole_true, borehole_failmodel_random
from surmise.emulation import emulator
from surmise.calibration import calibrator
from time import time


def alg(thetaprior, n=25, maxthetas=500, flag_failmodel=True, random_fail=False, obviate=True):
    if flag_failmodel is False:
        bh_model = borehole_model
    elif random_fail:
        bh_model = borehole_failmodel_random
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
    print(np.round(np.quantile(cal.theta.rnd(10000), (0.025, 0.5, 0.975), axis=0), 3))

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


#%% theta plots
import matplotlib.pyplot as plt

def plot_thetaprog(resdict):
    """Plot quantile progression."""

    n = len(resdict)
    nsim = np.array(resdict[0]['ncomp'])
    fig, ax = plt.subplots(1, 4, figsize=(12, 4), sharex=True, sharey=True)
    for i in range(n):
        for k in range(4):
            compress_quantiles = np.array(resdict[i]['quantile'])
            ax[k].plot(nsim, compress_quantiles[:, 0, k], linewidth=2, alpha=0.15)
            ax[k].plot(nsim, compress_quantiles[:, -1, k], linewidth=2, alpha=0.15)
            if i == 0:
                ax[k].plot(nsim, postthetarng[:, k][0] * np.ones(nsim.shape), color='b', linewidth=3, linestyle= '--')
                ax[k].plot(nsim, postthetarng[:, k][1] * np.ones(nsim.shape), color='b', linewidth=3)
                ax[k].plot(nsim, postthetarng[:, k][2] * np.ones(nsim.shape), color='b', linewidth=3, linestyle= '--')

                # ax[k].plot(nsim, 0.5*np.ones(nsim.shape), color='r', linewidth=3, linestyle= '--')
    for i, axi in enumerate(ax):
        axi.yaxis.set_tick_params(labelbottom=True)
        axi.xaxis.set_tick_params(labelbottom=True)
        axi.set_ylabel(r'$\theta_{:d}$'.format(i+1), rotation=0, size=15)
        # axi.set_ylim([0.45, 0.55])
    fig.text(0.5, -0.05, 'no. completed simulations', ha='center')
    plt.tight_layout()
    # plt.savefig('nofail_theta_prog.png', dpi=150)
    return

#%% time plots
def plot_time(resdict):
    """Plot calibration time."""
    n = len(resdict)
    nsim = np.array(resdict[0]['ncomp'])
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

    emutimes = np.array([resdict[i]['emutime'] for i in range(n)]).T
    emumean = emutimes.mean(1)
    emustd = emutimes.std(1)
    ax[0].plot(nsim, emutimes, linewidth=2)
    # ax[0].plot(nsim, np.maximum(0, emumean + sps.t.ppf(0.025, n-1) * emustd),
    #            color='k', linewidth=2, linestyle='--')
    # ax[0].plot(nsim, emumean + sps.t.ppf(0.975, n-1) * emustd,
    #            color='k', linewidth=2, linestyle='--')
    ax[0].set_title('emulator update')
    ax[0].set_ylabel('time (s)')

    caltimes = np.array([resdict[i]['caltime'] for i in range(n)]).T
    calmean = caltimes.mean(1)
    calstd = caltimes.std(1)
    ax[1].plot(nsim, caltimes, linewidth=1.5)
    # ax[1].plot(nsim, np.maximum(0.01, calmean + sps.t.ppf(0.025, n-1) * calstd),
               # color='k', linewidth=1.5, linestyle='--')
    # ax[1].plot(nsim, calmean + sps.t.ppf(0.975, n-1) * calstd,
               # color='k', linewidth=1.5, linestyle='--')
    ax[1].set_title('calibrator fit')
    ax[1].set_ylabel('time (s)')

    for i, axi in enumerate(ax):
        axi.yaxis.set_tick_params(labelbottom=True)
        axi.xaxis.set_tick_params(labelbottom=True)
    fig.text(0.5, -0.02, 'no. completed simulations', ha='center')
    plt.tight_layout()
    # plt.savefig('nofail_time.png', dpi=150)
    return



#%% alg loop time

def plot_looptime(resdict):
    """Plot calibration time."""
    n = len(resdict)
    nsim = np.array(resdict[0]['ncomp'])
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)

    looptimes = np.array([resdict[i]['looptime'] for i in range(n)]).T
    for i in range(n):
        looptimes[:, i] = np.ediff1d(looptimes[:, i], to_end=0)

    loopmean = looptimes.mean(1)
    loopstd = looptimes.std(1)
    ax.plot(nsim[:-1], loopmean[:-1], color='k', linewidth=2)
    ax.plot(nsim[:-1], np.maximum(0, loopmean + sps.t.ppf(0.025, n-1) * loopstd)[:-1],
               color='k', linewidth=2, linestyle='--')
    ax.plot(nsim[:-1], loopmean[:-1] + sps.t.ppf(0.975, n-1) * loopstd[:-1],
               color='k', linewidth=2, linestyle='--')
    ax.set_ylabel('time (s)')

    fig.text(0.5, -0.02, 'no. completed simulations', ha='center')
    plt.tight_layout()
    # plt.savefig('nofail_looptime.png', dpi=150)
    return

# %% theta pairplots from calibrated model
import seaborn as sns
import pandas as pd

def plot_thetapairs(cal):
    thetas = cal.theta.rnd(1000)
    df = pd.DataFrame(thetas)

    sns.reset_orig()
    sns.set_theme(style='white', font_scale=2)
    sns.set_palette('Dark2')

    g = sns.PairGrid(df,
                     layout_pad=0.005,
                     despine=False,
                     diag_sharey=False)

    g.map_diag(sns.kdeplot, linewidth=4, common_norm=False)
    g.map_offdiag(sns.histplot, bins=10, alpha=0.75)
    g.map_offdiag(sns.kdeplot, linewidths=3, levels=[0.05], alpha=0.6)

    # parameter labels
    for i in np.arange(4):
        ax = g.axes[i][i]
        ax.annotate(r'$\theta_{:d}$'.format(i+1), xy=(0.05, 0.75), size=36, xycoords=ax.transAxes)
        for j in np.arange(4):
            if i != j:
                ax = g.axes[i][j]
                sns.scatterplot(x=np.array((0.5, 0.5)), y=np.array((0.5, 0.5)), s=500, color='r', marker='X', ax=ax, legend=False, alpha=0.75)

    g.set(xlim=[0.1, 0.9], ylim=[0.1, 0.9])
    g.set(xticks=[0.3, 0.5, 0.7], yticks=[0.3, 0.5, 0.7])
    g.set(xlabel='', ylabel='')
    g.fig.subplots_adjust(wspace=.0, hspace=.0)
    # g.savefig('nofail_thetapairs.png', dpi=150)


# %% runs with failures
res_noobv = {'cal': [], 'emu': [], 'res': []}
for i in np.arange(10):
    cal_fail, emu_fail, res_fail = alg(thetaprior, maxthetas=120, flag_failmodel=True, random_fail=False, obviate=True)
    res_noobv['cal'].append(cal_fail)
    res_noobv['emu'].append(emu_fail)
    res_noobv['res'].append(res_fail)


# %% one-shop plots

print('obviation=True, random=False')

resultlist = res_noobv

sns.reset_defaults()
plot_thetaprog(resultlist['res'])

plot_time(resultlist['res'])

plot_looptime(resultlist['res'])

plot_thetapairs(resultlist['cal'][0])

#%% profiling one loop
# import cProfile
# import pstats
# import io

# pr = cProfile.Profile()
# pr.enable()

# res = alg(thetaprior, maxthetas=200, flag_failmodel=False)
# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
# ps.print_stats()

# with open('alg_profile.txt', 'w+') as f:
#     f.write(s.getvalue())
