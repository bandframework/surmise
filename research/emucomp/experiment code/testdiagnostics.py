import numpy as np
from scipy.stats import norm


def rmse(emu, x, theta, testf):
    return np.sqrt(np.nanmean((emu.predict(x, theta).mean() - testf) ** 2))


def mae(emu, x, theta, testf):
    return np.nanmean(np.abs(emu.predict(x, theta).mean() - testf))


def medae(emu, x, theta, testf):
    return np.nanmedian(np.abs(emu.predict(x, theta).mean() - testf))


def me(emu, x, theta, testf):
    return np.nanmean(emu.predict(x, theta).mean() - testf)


def interval_stats(emu, x, theta, testf, pr=0.9):
    p = emu.predict(x, theta)
    mean = p.mean()
    stdev = np.sqrt(p.var())

    alph = 1 - pr

    f = testf
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


def crps(emu, x, theta, testf):
    p = emu.predict(x, theta)
    mean = p.mean()
    var = p.var()
    stdev = np.sqrt(var) + 10 ** (-12)

    z = (testf - mean) / stdev

    crpss = -stdev * (z * (2*norm.cdf(z) - 1) + 2*norm.ppf(z) - 1 / np.sqrt(np.pi))
    return np.nanmean(crpss)


def errors(x, testtheta, model, modelname, random, ntheta,
           failfraction, emu=None, emutime=None, method=None):
    results = {}
    results['method'] = method
    results['function'] = modelname
    results['randomfailures'] = str(random)
    results['failfraction'] = failfraction
    results['nx'] = x.shape[0]
    results['n'] = ntheta

    if emu is not None:
        if 'logvarc' in emu._info.keys():
            results['dampalpha'] = emu._info['dampalpha']
            results['avgvarconstant'] = np.mean(np.exp(emu._info['logvarc']))
            # print(emu._info['logvarc'])
            # print(np.median(emu._info['gvar']), np.max(emu._info['gvar']), np.min(emu._info['gvar']))
            results['varc_status'] = emu._info['varc_status']
        else:
            results['dampalpha'] = np.nan
            results['avgvarconstant'] = np.nan
            results['varc_status'] = 'n/a'

        testf = model(x, testtheta, failfraction)
        frng = np.nanmax(testf) - np.nanmin(testf)

        results['rmse'] = rmse(emu, x, testtheta, testf) / frng
        results['mae'] = mae(emu, x, testtheta, testf) / frng
        results['medae'] = medae(emu, x, testtheta, testf) / frng
        results['me'] = me(emu, x, testtheta, testf) / frng
        results['crps'] = crps(emu, x, testtheta, testf)
        int_stats = interval_stats(emu, x, testtheta, testf)
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


def errors_fayans(x, testtheta, testf, modelname, ntheta,
                  emu=None, emutime=None, method=None):
    results = {}
    results['method'] = method
    results['function'] = modelname
    results['randomfailures'] = None
    results['failfraction'] = None
    results['nx'] = x.shape[0]
    results['n'] = ntheta

    if emu is not None:
        if 'logvarc' in emu._info.keys():
            results['dampalpha'] = emu._info['dampalpha']
            results['avgvarconstant'] = np.mean(np.exp(emu._info['logvarc']))
            # print(emu._info['logvarc'])
            # print(np.median(emu._info['gvar']), np.max(emu._info['gvar']), np.min(emu._info['gvar']))
            results['varc_status'] = emu._info['varc_status']
        else:
            results['dampalpha'] = np.nan
            results['avgvarconstant'] = np.nan
            results['varc_status'] = 'n/a'

        frng = np.nanmax(testf) - np.nanmin(testf)

        results['rmse'] = rmse(emu, x, testtheta, testf) / frng
        results['mae'] = mae(emu, x, testtheta, testf) / frng
        results['medae'] = medae(emu, x, testtheta, testf) / frng
        results['me'] = me(emu, x, testtheta, testf) / frng
        results['crps'] = crps(emu, x, testtheta, testf)
        int_stats = interval_stats(emu, x, testtheta, testf)
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
