import numpy as np
import scipy.stats as sps
from surmise.emulation import emulator
from surmise.calibration import calibrator
import matplotlib.pyplot as plt

fulldata = np.load('fayansdata/newp2.npy', allow_pickle=True).item() #

theta = fulldata['t']
feval = fulldata['f']
x = fulldata['x']
failval = fulldata['e']
feval[failval > 0.5] = np.nan

# subset without nan theta
nantheta = np.isnan(theta).sum(1) > 0.5
print('number of parameters containing nan: {:d}'.format(nantheta.sum()))
theta = theta[~nantheta]
feval = feval[~nantheta]
failval = failval[~nantheta]

# subset by l1-norm criterion in theta
thetanorm1 = np.abs(theta - 0.5).sum(1)
outofrangetheta = thetanorm1 >= 2
print('number of out-of-range parameters: {:d}'.format(outofrangetheta.sum()))
theta = theta[~outofrangetheta]
feval = feval[~outofrangetheta]
failval = failval[~outofrangetheta]

# subset by f values
fevaltoobig = np.any(np.abs(feval) > 15, 1)
print('number of parameters with large fs: {:d}'.format(fevaltoobig.sum()))
theta = theta[~fevaltoobig]
feval = feval[~fevaltoobig]
failval = failval[~fevaltoobig]

# subset by theta values
thetatoobig = np.any(np.abs(theta) > 2, 1)
print('number of parameters with large thetas: {:d}'.format(thetatoobig.sum()))
theta = theta[~thetatoobig]
feval = feval[~thetatoobig]
failval = failval[~thetatoobig]

print('max theta component = {:.6f}'.format(theta.max()))

# subset with less failures
toomuchfail = failval.sum(1) > 50
print('number of discarded parameters: {:d}'.format(toomuchfail.sum()))

subtheta = theta[~toomuchfail]
subfeval = feval[~toomuchfail]
subfailval = failval[~toomuchfail]

# shape of remaining parameters
print('number of remaining parameters: {:d}'.format(theta.shape[0]))


#%% Prior
class thetaprior:
    """ This defines the class instance of priors provided to the methods. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            logprior = -50 * np.sum((theta - 0.6) ** 2, axis=1)
            logprior += 2*np.log(2-np.sum(np.abs(theta - 0.5), axis=1))
            flag = np.sum(np.abs(theta - 0.5), axis=1) > 2
            logprior[flag] = -np.inf
            logprior = np.array(logprior,ndmin=1)
        else:
            logprior = -50 * np.sum((theta - 0.6) ** 2)
            logprior += 2*np.log(2-np.sum(np.abs(theta - 0.5)))
            if np.sum(np.abs(theta - 0.5)) > 2:
                logprior = -np.inf
            logprior = np.array(logprior,ndmin=1)
        return logprior.reshape((len(theta), 1))
    def rnd(n):
        if n > 1:
            rndval = np.vstack((sps.norm.rvs(0.6, 0.1, size=(n,13))))
            flag = np.sum(np.abs(rndval - 0.5), axis=1) > 2
            while np.any(flag):
                rndval[flag,:] = np.vstack((sps.norm.rvs(0.6, 0.1, size=(np.sum(flag),13))))
                flag = np.sum(np.abs(rndval - 0.5), axis=1) > 2
        else:
            rndval = sps.norm.rvs(0.6, 0.1, size =13)
            while np.sum(np.abs(rndval - 0.5)) > 2:
                rndval = np.vstack((sps.norm.rvs(0.6, 0.1,size = 13)))
        return rndval.reshape((n,13))


#%% data for emu-cal
selectedindex = np.random.choice(np.arange(subtheta.shape[0]), replace=False, size=500)

subtheta = subtheta[selectedindex]
subfeval = subfeval[selectedindex]
y = np.zeros(x.shape[0])
yvar = np.ones(x.shape[0])

emu = emulator(x=x, theta=subtheta, f=subfeval, method = 'PCGPwM',
               options={'xrmnan': 'all',
                        'thetarmnan': 'never',
                        'return_grad': True})

cal = calibrator(emu, y, x, thetaprior, yvar, method='directbayeswoodbury')


#%% plot prescreened thetas with failures
import seaborn as sns
import pandas as pd

fayans_cols = [r'$\rho_{\mathrm{eq}}$', r'$E/A$', r'$K$', r'$J$',
                    r'$L$', '$h^{\mathrm{v}}_{2{-}}$',
                    r'$a^{\mathrm{s}}_{+}$',
                    r'$h^{\mathrm{s}}_{\nabla}$',
                    r'$\kappa$', r'$\kappa^\prime$',
                    r'$f^{\xi}_{\mathrm{ex}}$',
                    r'$h^{\xi}_{+}$', r'$h^{\xi}_{\nabla}$']


def plot_allthetas(allthetas, postthetas=None, sumfails=None):
    toplotdf = allthetas
    toplotdf['sumfails'] = sumfails
    sns.reset_orig()
    sns.set_theme(style='white', font_scale=3)
    sns.set_palette('cubehelix')

    step = 10000

    g = sns.PairGrid(toplotdf, hue='sumfails', # hue='cat',
                     layout_pad=0.005,
                     palette="RdYlBu_r",
                     despine=False,
                     diag_sharey=False)

    # parameter labels
    for i in np.arange(len(fayans_cols)):
        ax = g.axes[i][i]
        ax.annotate(fayans_cols[i], xy=(0.3, 0.3), size=52, xycoords=ax.transAxes)

    g.map_offdiag(sns.scatterplot, alpha=0.2)
    g.set(xlabel='', ylabel='')

    cax = g.fig.add_axes([.94, .3, .02, .4])
    cbar = g.fig.colorbar(mappable=plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=np.min(sumfails), vmax=np.max(sumfails))),
                          cax=cax, pad=0.05)
    g.fig.subplots_adjust(wspace=.0, hspace=.0, right=0.92)
    cbar.ax.set_title('#fail', y=1.05)


#%% function that plots from calibrator
def plot_thetapairs(cal):
    thetas = cal.theta.rnd(200)
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
    for i in np.arange(len(fayans_cols)):
        ax = g.axes[i][i]
        ax.annotate(fayans_cols[i], xy=(0.05, 0.75), size=36, xycoords=ax.transAxes)
        # for j in np.arange(4):
        #     if i != j:
        #         ax = g.axes[i][j]
        #         sns.scatterplot(x=np.array((0.5, 0.5)), y=np.array((0.5, 0.5)), s=500, color='r', marker='X', ax=ax, legend=False, alpha=0.75)

    g.set(xlim=[0.1, 0.9], ylim=[0.1, 0.9])
    g.set(xticks=[0.3, 0.5, 0.7], yticks=[0.3, 0.5, 0.7])
    g.set(xlabel='', ylabel='')
    g.fig.subplots_adjust(wspace=.0, hspace=.0)


#%% plotting

import time
# # subset theta further if desired
# t1 = time.time()
# plot_allthetas(pd.DataFrame(theta), sumfails=failval.sum(1))
# print('plot time: {:.2f} min'.format((time.time() - t1) / 60))

plot_thetapairs(cal)