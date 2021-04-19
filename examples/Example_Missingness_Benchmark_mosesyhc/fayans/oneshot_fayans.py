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

# dataset with less failures
toomuchfail = failval.sum(1) > 50
print('number of discarded parameters: {:d}'.format(toomuchfail.sum()))

theta = theta[~toomuchfail]
feval = feval[~toomuchfail]
failval = failval[~toomuchfail]

selectedindex = np.random.choice(np.arange(theta.shape[0]), replace=False, size=500)

subtheta= theta[selectedindex]
subfeval = feval[selectedindex]
y = np.zeros(x.shape[0])
yvar = np.ones(x.shape[0])


#%% plot thetas
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
        ax.annotate(fayans_cols[i], xy=(0.35, 0.35), size=52, xycoords=ax.transAxes)

    g.map_offdiag(sns.scatterplot, alpha=0.2)
    g.set(xlabel='', ylabel='')

    cax = g.fig.add_axes([.94, .3, .02, .4])
    cbar = g.fig.colorbar(mappable=plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=np.min(sumfails), vmax=np.max(sumfails))),
                          cax=cax, pad=0.05)
    g.fig.subplots_adjust(wspace=.0, hspace=.0, right=0.92)
    cbar.ax.set_title('#fail', y=1.05)

# plot_allthetas(pd.DataFrame(theta), sumfails=failval.sum(1))