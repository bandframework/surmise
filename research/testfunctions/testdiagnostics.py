import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def rmse(emu, x, theta, model):
    return np.sqrt(np.nanmean((emu.predict(x, theta).mean() - model(x, theta)) ** 2))


def mae(emu, x, theta, model):
    return np.nanmean(np.abs(emu.predict(x, theta).mean() - model(x, theta)))


def medae(emu, x, theta, model):
    return np.nanmedian(np.abs(emu.predict(x, theta).mean() - model(x, theta)))


def me(emu, x, theta, model):
    return np.nanmean(emu.predict(x, theta).mean() - model(x, theta))


def interval_stats(emu, x, theta, model, pr=0.9):
    p = emu.predict(x, theta)
    mean = p.mean()
    stdev = np.sqrt(p.var())

    alph = 1 - pr

    f = model(x, theta)
    mask = ~np.isnan(f)

    ci = np.zeros((2, *mean.shape))
    ci[0] = mean + norm.ppf(alph/2) * stdev
    ci[1] = mean + norm.ppf(1 - alph/2) * stdev

    under = np.less(f[mask], ci[0][mask])
    over = np.greater(f[mask], ci[1][mask])

    coverage = (1 - np.logical_or(under, over)).mean()
    avgintwidth = (ci[1] - ci[0]).mean()
    intScore = np.nanmean((ci[1] - ci[0])[mask] +
                2/alph * (ci[0] - f)[mask] * under +
               2/alph * (f - ci[1])[mask] * over)
    return coverage, avgintwidth, intScore


def crps(emu, x, theta, model):
    p = emu.predict(x, theta)
    mean = p.mean()
    var = p.var()
    stdev = np.sqrt(var) + 10 ** (-12)

    z = (model(x, theta) - mean) / stdev

    crpss = -stdev * (z * (2*norm.cdf(z) - 1) + 2*norm.ppf(z) - 1 / np.sqrt(np.pi))
    return np.nanmean(crpss)


def plot_fails(x, theta, model):
    f = model(x, theta)
    fisnan = np.isnan(f).sum(0)
    # fisnan = np.isinf(f).sum(0)
    df = pd.DataFrame(theta)
    df['fails'] = fisnan.astype(float)
    df['failsbin'] = fisnan > 0
    # sns.pairplot(df, vars=df.columns[:theta.shape[1]], hue='fails', palette='coolwarm')
    colors = ["#4374B3", "#FF0B04"]
    sns.set_palette(sns.color_palette(colors))
    del df['fails']
    meltdf = pd.melt(df, df.columns[-1], df.columns[:-1])
    g = sns.FacetGrid(meltdf, col="variable", hue='failsbin', legend_out=True)
    g.map(sns.kdeplot, "value", shade=True)

    for i, ax in enumerate(g.axes[0]):
        ax.set_xlabel(r'$\theta_{:d}$'.format(i + 1))
        ax.set_title('')

    g.add_legend()
    g._legend.set_title('Fails')
    new_labels = ['0', '> 0']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)


def plot_marginal(x, theta, model, modelname, emus=()):
    sns.set_palette('icefire')
    markers = ['o', 'v', 's', 'X', 'p', 'D', '*', '+']
    if theta.shape[0] > 100:
        theta = theta[:100, :]

    alpha = emus[-1]._info['dampalpha']

    f = model(x, theta)

    fig, axes = plt.subplots(1, theta.shape[1], figsize=(4 * theta.shape[1], 4))
    for k, ax in enumerate(axes):
        sns.scatterplot(x=theta[:, k], y=f[10], color='r', label='true', marker='D', ax=ax, s=20, zorder=100)
        for i, emu in enumerate(emus):
            emumean = emu.predict(x, theta).mean()[10]
            emulabel = emu.method.__name__.split('.')[-1]
            if 'logvarc' in emu._info:
                emulabel = r'$\beta$=' + str(np.round(np.mean(emu._info['logvarc']), 0))
            sns.lineplot(theta[:, k], emumean, label=emulabel, marker=markers[i], sort=True, ax=ax, zorder=1)
        ax.set_xlabel(r'$\theta_{:d}$'.format(k + 1))
        ax.set_ylabel('function prediction')

    plt.suptitle(r'{:s}, $\alpha$ = {:.3f}'.format(modelname, alpha))
    plt.legend()
    plt.tight_layout()

    dirname = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\figs'
    fname = r'\{:s}_alph{:.3f}.png'.format(modelname, alpha)
    plt.savefig(dirname + fname, dpi=75)


def errors(x, testtheta, model, modelname, random, ntheta, emu=None, emutime=None, method=None):
    results = {}
    results['method'] = method
    results['function'] = modelname
    results['randomfailures'] = random
    results['nx'] = x.shape[0]
    results['n'] = ntheta

    if emu is not None:
        if 'logvarc' in emu._info.keys():
            results['dampalpha'] = emu._info['dampalpha']
            results['avgvarconstant'] = '{:.3f}'.format(np.mean(np.exp(emu._info['logvarc'])))
            # print(emu._info['logvarc'])
            # print(np.median(emu._info['gvar']), np.max(emu._info['gvar']), np.min(emu._info['gvar']))
            results['varc_status'] = emu._info['varc_status']
        else:
            results['dampalpha'] = np.nan
            results['avgvarconstant'] = np.nan
            results['varc_status'] = 'n/a'

        fstd = np.nanstd(model(x, testtheta))

        results['rmse'] = '{:.3f}'.format(rmse(emu, x, testtheta, model) / fstd)
        results['mae'] = '{:.3f}'.format(mae(emu, x, testtheta, model) / fstd)
        results['medae'] = '{:.3f}'.format(medae(emu, x, testtheta, model) / fstd)
        results['me'] = '{:.3f}'.format(me(emu, x, testtheta, model) / fstd)
        results['crps'] = '{:.3f}'.format(crps(emu, x, testtheta, model) / fstd)
        int_stats = interval_stats(emu, x, testtheta, model)
        results['coverage'] = int_stats[0]
        results['avgintwidth'] = int_stats[1]
        results['intscore'] = int_stats[2]
    else:
        results['dampalpha'] = np.nan
        results['avgvarconstant'] = np.nan
        results['varc_status'] = 'n/a'
        results['rmse'] = np.nan
        results['mae'] = np.nan
        results['medae'] = np.nan
        results['me'] = np.nan
        results['crps'] = np.nan
        results['coverage'] = np.nan
        results['avgintwidth'] = np.nan
        results['intscore'] = np.nan

    if emutime is not None:
        results['emutime'] = emutime
    else:
        results['emutime'] = np.nan
    return results
