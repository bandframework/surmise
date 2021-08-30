import numpy as np
from scipy.stats import norm


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
