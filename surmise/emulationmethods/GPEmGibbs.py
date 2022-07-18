""" """
import warnings
import numpy as np
import scipy.optimize as spo
import scipy.linalg as spla
from surmise.emulationsupport.GPEmGibbs_covmat_helper import setxcovf, setthetacovf


def fit(fitinfo, x, theta, f, misval=None, cat=False,
        xcovfname='exp', thetacovfname='exp', **kwargs):
    '''

    '''
    f = f.T

    fitinfo['theta'] = theta
    fitinfo['x'] = x
    fitinfo['f'] = f
    fitinfo['cat'] = cat
    fitinfo['xcovfname'] = xcovfname
    fitinfo['thetacovfname'] = thetacovfname
    if misval is not None:
        if np.isnan(f).any() and np.any(misval != np.isnan(f)):
            misval = misval ^ np.isnan(f)
            warnings.warn('''The provided missing value matrix (mis) 
            is updated to include NaN values.''')
    else:
        misval = np.isnan(f)

    emuinfo = __initialize(fitinfo, misval)

    emulation_hypest(emuinfo, 'indp')

    if emuinfo['misbool']:
        Rs, Cs = emulation_matrixblockerfunction(emuinfo)
        emulation_hypest(emuinfo)
        emulation_imputeiter(emuinfo, Rs, Cs, tol=10 ** (-3))
        emulation_hypest(emuinfo)
        emulation_imputeiter(emuinfo, Rs, Cs)
        emulation_hypest(emuinfo)
        # emulation_imputeiter(emuinfo, Rs, Cs, tol=10 ** (-7))
        # emulation_hypest(emuinfo)
    else:
        emulation_hypest(emuinfo)

    fitinfo['emuinfo'] = emuinfo
    return


def predict(predinfo, fitinfo, x, theta, **kwargs):
    emuinfo = fitinfo['emuinfo']

    r = emuinfo['covthetaf'](theta, emuinfo['theta'], emuinfo['gammathetacovhyp'])

    yhat = emuinfo['mu'] + r @ emuinfo['residinv']

    r0 = np.diag(emuinfo['covthetaf'](theta, theta, emuinfo['gammathetacovhyp']))

    if fitinfo['cat']:
        xval = x[:, :-1].astype(float)
        xcat = x[:, -1]
        s0 = np.diag(emuinfo['covxf'](xval, xval, emuinfo['gammaxcovhyp'], type1=xcat, type2=xcat))
        s = emuinfo['covxf'](xval, emuinfo['xval'], emuinfo['gammaxcovhyp'], type1=xcat, type2=emuinfo['xcat'])
    else:
        xval = x
        s0 = np.diag(emuinfo['covxf'](xval, xval, emuinfo['gammaxcovhyp']))
        s = emuinfo['covxf'](xval, emuinfo['xval'], emuinfo['gammaxcovhyp'])
    varhatR = r0 - np.diag(r @ emuinfo['R_inv'] @ r.T)
    varhatS = s0 - np.diag(s @ emuinfo['S_inv'] @ s.T)

    predinfo['mean'] = yhat
    predinfo['varR'] = varhatR
    predinfo['varS'] = varhatS
    predinfo['var'] = np.outer(varhatR, varhatS)

    return


def __initialize(fitinfo, misval):
    emuinfo = {}
    f = fitinfo['f']
    x = fitinfo['x']
    theta = fitinfo['theta']
    n, m = f.shape
    if fitinfo['cat']:
        xval = (x[:, :-1]).astype(float)
        xcat = x[:, -1]
    else:
        xval = x
        xcat = np.ones(m).astype(int)

    uniquecat = np.unique(xcat)
    numcat = uniquecat.shape[0]

    emuinfo['nu'] = np.max((m + 60,
                            3 * m,
                            0.8 * n)).astype('int')
    emuinfo['f'] = f
    emuinfo['theta'] = theta
    emuinfo['xcat'] = xcat
    emuinfo['xval'] = xval
    emuinfo['uniquecat'] = uniquecat
    emuinfo['numcat'] = numcat
    emuinfo['n'] = n
    emuinfo['m'] = m

    emuinfo['misval'] = misval
    emuinfo['misbool'] = misval.any()
    # emuinfo['blocking'] = 'individual'
    # emuinfo['modeltype'] = 'parasep'

    setxcovf(emuinfo, fitinfo['xcovfname'])
    setthetacovf(emuinfo, fitinfo['thetacovfname'])

    __sethyp(emuinfo)

    return emuinfo


def predictmean(predinfo, **kwargs):
    return predinfo['mean']


def predictvar(predinfo, **kwargs):
    return predinfo['var']


def emulation_hypest(emuinfo, modeltype='parasep'):
    if modeltype == 'parasep':
        try:
            likeattempt = emulation_lik_parasep(emuinfo['gammahat'], emuinfo)
            if np.isinf(likeattempt):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_parasep(emuinfo['gammahat'], emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt), 1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_parasep(emuinfo['gammahat'], emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt), 1))
        bounds = spo.Bounds(emuinfo['gammaLB'], emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_parasep, emuinfo['gammahat'],
                             args=(emuinfo),
                             method='L-BFGS-B',
                             options={'disp': False, 'ftol': myftol},
                             jac=emulation_dlik_parasep,
                             bounds=bounds)
        emuinfo['gammahat'] = opval.x
        emuinfo['R'] = emulation_getR(emuinfo)
        emuinfo['mu'] = emulation_getmu(emuinfo)

        emuinfo['gammathetacovhyp'] = emuinfo['gammahat'][0:(emuinfo['hypstatparstructure'][0])]
        emuinfo['gammaxcovhyp'] = emuinfo['gammahat'][
                                  (emuinfo['hypstatparstructure'][0]):(emuinfo['hypstatparstructure'][1])]
        emuinfo['gammasigmasq'] = np.exp(
            emuinfo['gammahat'][(emuinfo['hypstatparstructure'][1]):(emuinfo['hypstatparstructure'][2])])
        emuinfo['gammamu'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][2]):]

        sigma2 = emulation_getsigma2(emuinfo)
        Sigmapart1 = emulation_getS(emuinfo)

        emuinfo['S'] = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
        emuinfo['R_chol'], _ = spla.lapack.dpotrf(emuinfo['R'], True, True)

        emuinfo['R_inv'] = spla.solve_triangular(emuinfo['R_chol'],
                                                 spla.solve_triangular(emuinfo['R_chol'], np.eye(emuinfo['n']),
                                                                       lower=True), lower=True, trans=True)

        resid = emuinfo['fpred'] - emuinfo['mu']
        residhalf = spla.solve_triangular(emuinfo['R_chol'], resid, lower=True)
        emuinfo['residinv'] = spla.solve_triangular(emuinfo['R_chol'], residhalf, lower=True, trans=True)

        emuinfo['S_chol'], _ = spla.lapack.dpotrf(emuinfo['S'], True, True)
        emuinfo['S_inv'] = spla.solve_triangular(emuinfo['S_chol'],
                                                 spla.solve_triangular(emuinfo['S_chol'], np.eye(emuinfo['m']),
                                                                       lower=True), lower=True, trans=True)

        emuinfo['pw'] = np.linalg.solve(emuinfo['S_chol'].T,
                                        np.linalg.solve(emuinfo['S_chol'], emuinfo['residinv'].T))
    else:
        try:
            likeattempt = emulation_lik_indp(emuinfo['gammahat'], emuinfo)
            if np.isinf(likeattempt):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_indp(emuinfo['gammahat'], emuinfo)
            myftol = 0.25 / np.max((np.abs(likeattempt), 1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_indp(emuinfo['gammahat'], emuinfo)
            myftol = 0.25 / np.max((np.abs(likeattempt), 1))

        bounds = spo.Bounds(emuinfo['gammaLB'], emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_indp, emuinfo['gammahat'],
                             args=(emuinfo),
                             method='L-BFGS-B',
                             options={'disp': False, 'ftol': myftol},
                             jac=emulation_dlik_indp,
                             bounds=bounds)
        emuinfo['gammahat'] = opval.x
        emuinfo['R'] = emulation_getR(emuinfo)
        emuinfo['mu'] = emulation_getmu(emuinfo)
        resid = emuinfo['f'] - emuinfo['mu']
        mu = emuinfo['mu']
        if emuinfo['misbool']:
            fpred = np.ones(emuinfo['f'].shape)
            donealr = np.zeros(emuinfo['m'])
            for k2 in range(0, emuinfo['m']):
                inds = (np.where(emuinfo['misval'][:, k2] < 0.5)[0]).astype(int)
                ninds = (np.where(emuinfo['misval'][:, k2] > 0.5)[0]).astype(int)
                Robs = emuinfo['R'][np.ix_(inds, inds)]
                Rnobsobs = emuinfo['R'][np.ix_(ninds, inds)]
                cholRobs, _ = spla.lapack.dpotrf(Robs, True, True)
                for k in range(k2, emuinfo['m']):
                    if (np.sum(np.abs(emuinfo['misval'][:, k] ^ emuinfo['misval'][:, k2])) < 0.5) and (
                            donealr[k] < 0.5):
                        resvalhalf = spla.solve_triangular(cholRobs, resid[inds, k], lower=True)
                        resvalinv = spla.solve_triangular(cholRobs, resvalhalf, lower=True, trans=True)
                        fpred[ninds, k] = emuinfo['mu'][k] + np.matmul(Rnobsobs, resvalinv)
                        fpred[inds, k] = emuinfo['f'][inds, k]
                        donealr[k] = 1
            emuinfo['fpred'] = fpred[:]
        else:
            emuinfo['fpred'] = emuinfo['f'][:]
    return


def __sethyp(emuinfo):
    numcat = emuinfo['numcat']
    f = emuinfo['f']
    xcat = emuinfo['xcat']
    uniquecat = emuinfo['uniquecat']

    gammasigmasq0 = np.ones(numcat)
    gammasigmasqLB = np.ones(numcat)
    gammasigmasqUB = np.ones(numcat)
    gammamu0 = np.ones(numcat)
    gammamuLB = np.ones(numcat)
    gammamuUB = np.ones(numcat)

    for k in range(numcat):
        fconsid = f[np.squeeze(np.where(xcat == uniquecat[k]))]
        meanvalsnow = np.nanmean(fconsid, 1)
        gammasigmasq0[k] = 2 + np.log(np.nanvar(fconsid))
        gammasigmasqLB[k] = 0 + gammasigmasq0[k]
        gammasigmasqUB[k] = 8 + gammasigmasq0[k]
        gammamu0[k] = np.nanmean(meanvalsnow)
        gammamuLB[k] = np.nanmin(meanvalsnow) - (np.nanmax(meanvalsnow) - np.nanmin(meanvalsnow))
        gammamuUB[k] = np.nanmax(meanvalsnow) + (np.nanmax(meanvalsnow) - np.nanmin(meanvalsnow))

    numthetahyp = emuinfo['gammathetacovhyp0'].shape[0]
    numxhyp = emuinfo['gammaxcovhyp0'].shape[0]
    hypstatparstructure = [numthetahyp,
                           numthetahyp + numxhyp,
                           numthetahyp + numxhyp + numcat,
                           numthetahyp + numxhyp + numcat + numcat]
    emuinfo['gamma0'] = np.concatenate(
        (emuinfo['gammathetacovhyp0'], emuinfo['gammaxcovhyp0'], gammasigmasq0, gammamu0))
    emuinfo['gammaLB'] = np.concatenate(
        (emuinfo['gammathetacovhypLB'], emuinfo['gammaxcovhypLB'], gammasigmasqLB, gammamuLB))
    emuinfo['gammaUB'] = np.concatenate(
        (emuinfo['gammathetacovhypUB'], emuinfo['gammaxcovhypUB'], gammasigmasqUB, gammamuUB))
    emuinfo['hypstatparstructure'] = hypstatparstructure
    emuinfo['gammahat'] = emuinfo['gamma0']
    return


def __hypest(emuinfo, modeltype=None):
    if modeltype is None:
        modeltype = emuinfo['modeltype']
    if modeltype == 'parasep':
        try:
            likeattempt = emulation_lik_parasep(emuinfo['gammahat'], emuinfo)
            if np.isinf(likeattempt):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_parasep(emuinfo['gammahat'], emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt), 1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_parasep(emuinfo['gammahat'], emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt), 1))
        bounds = spo.Bounds(emuinfo['gammaLB'], emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_parasep, emuinfo['gammahat'],
                             args=emuinfo,
                             method='L-BFGS-B',
                             options={'disp': False, 'ftol': myftol},
                             jac=emulation_dlik_parasep,
                             bounds=bounds)
        emuinfo['gammahat'] = opval.x
        emuinfo['R'] = emulation_getR(emuinfo)
        emuinfo['mu'] = emulation_getmu(emuinfo)

        emuinfo['gammathetacovhyp'] = emuinfo['gammahat'][0:(emuinfo['hypstatparstructure'][0])]
        emuinfo['gammaxcovhyp'] = emuinfo['gammahat'][
                                  (emuinfo['hypstatparstructure'][0]):(emuinfo['hypstatparstructure'][1])]
        emuinfo['gammasigmasq'] = np.exp(
            emuinfo['gammahat'][(emuinfo['hypstatparstructure'][1]):(emuinfo['hypstatparstructure'][2])])
        emuinfo['gammamu'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][2]):]

        sigma2 = emulation_getsigma2(emuinfo)

        Sigmapart1 = emulation_getS(emuinfo)
        emuinfo['S'] = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
        emuinfo['R_chol'], _ = spla.lapack.dpotrf(emuinfo['R'], True, True)

        emuinfo['R_inv'] = spla.solve_triangular(emuinfo['R_chol'], spla.solve_triangular(emuinfo['R_chol'], np.diag(
            np.ones(emuinfo['n'])), lower=True), lower=True, trans=True)

        resid = emuinfo['fpred'] - emuinfo['mu']
        residhalf = spla.solve_triangular(emuinfo['R_chol'], resid, lower=True)
        emuinfo['residinv'] = spla.solve_triangular(emuinfo['R_chol'], residhalf, lower=True, trans=True)

        emuinfo['S_chol'], _ = spla.lapack.dpotrf(emuinfo['S'], True, True)
        emuinfo['S_inv'] = spla.solve_triangular(emuinfo['S_chol'], spla.solve_triangular(emuinfo['S_chol'], np.diag(
            np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)

        emuinfo['pw'] = np.linalg.solve(emuinfo['S_chol'].T,
                                        np.linalg.solve(emuinfo['S_chol'], emuinfo['residinv'].T))
    elif modeltype == 'indp':
        try:
            likeattempt = emulation_lik_indp(emuinfo['gammahat'], emuinfo)
            if np.isinf(likeattempt):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_indp(emuinfo['gammahat'], emuinfo)
            myftol = 0.25 / np.max((np.abs(likeattempt), 1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_indp(emuinfo['gammahat'], emuinfo)
            myftol = 0.25 / np.max((np.abs(likeattempt), 1))

        bounds = spo.Bounds(emuinfo['gammaLB'], emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_indp, emuinfo['gammahat'],
                             args=(emuinfo),
                             method='L-BFGS-B',
                             options={'disp': False, 'ftol': myftol},
                             jac=emulation_dlik_indp,
                             bounds=bounds)
        emuinfo['gammahat'] = opval.x
        emuinfo['R'] = emulation_getR(emuinfo)
        emuinfo['mu'] = emulation_getmu(emuinfo)
        resid = emuinfo['f'] - emuinfo['mu']
        mu = emuinfo['mu']
        if emuinfo['misbool']:
            fpred = np.ones(emuinfo['f'].shape)
            donealr = np.zeros(emuinfo['m'])
            for k2 in range(0, emuinfo['m']):
                inds = (np.where(emuinfo['misval'][:, k2] < 0.5)[0]).astype(int)
                ninds = (np.where(emuinfo['misval'][:, k2] > 0.5)[0]).astype(int)
                Robs = emuinfo['R'][np.ix_(inds, inds)]
                Rnobsobs = emuinfo['R'][np.ix_(ninds, inds)]
                cholRobs, _ = spla.lapack.dpotrf(Robs, True, True)
                for k in range(k2, emuinfo['m']):
                    if (np.sum(np.abs(emuinfo['misval'][:, k] - emuinfo['misval'][:, k2])) < 0.5) and (
                            donealr[k] < 0.5):
                        resvalhalf = spla.solve_triangular(cholRobs, resid[inds, k], lower=True)
                        resvalinv = spla.solve_triangular(cholRobs, resvalhalf, lower=True, trans=True)
                        fpred[ninds, k] = emuinfo['mu'][k] + np.matmul(Rnobsobs, resvalinv)
                        fpred[inds, k] = emuinfo['f'][inds, k]
                        donealr[k] = 1
            emuinfo['fpred'] = fpred[:]
        else:
            emuinfo['fpred'] = emuinfo['f'][:]
    return


def emulation_lik_indp(gammav, emuinfo, fval=None):
    R = emulation_getR(emuinfo, gammav, False)
    (cholR, pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        warnings.warn('Cholesky decomposition encounters a near-singular matrix', stacklevel=2)
        return np.inf

    mu = emulation_getmu(emuinfo, gammav)
    sigma2 = emulation_getsigma2(emuinfo, gammav)
    if fval is None:
        fval = emuinfo['f']
    resid = fval - mu
    missval = emuinfo['misval']

    logs2sum = 0
    logdetRsum = 0
    donealr = np.zeros(emuinfo['m'])
    for k2 in range(0, emuinfo['m']):
        nhere = np.sum(1 - missval[:, k2])
        inds = (np.where(missval[:, k2] < 0.5)[0]).astype(int)
        Robs = R[np.ix_(inds, inds)]

        (cholRobs, pd) = spla.lapack.dpotrf(Robs, True, True)
        if pd > 0.5:
            warnings.warn('Cholesky decomposition encounters a near-singular matrix', stacklevel=2)
            return np.inf
        logdetval = 2 * np.sum(np.log(np.diag(cholRobs)))
        for k in range(k2, emuinfo['m']):
            if (np.sum(np.abs(missval[:, k] ^ missval[:, k2])) < 0.5) and (donealr[k] < 0.5):
                resvalhalf = spla.solve_triangular(cholRobs, resid[inds, k], lower=True)
                logs2sum += (nhere / 2 + emuinfo['nu']) * np.log(
                    1 / 2 * np.sum(resvalhalf * resvalhalf) + emuinfo['nu'] * sigma2[k]) - emuinfo['nu'] * np.log(
                    emuinfo['nu'] * sigma2[k])
                logdetRsum += logdetval
                donealr[k] = 1
    gammanorm = (gammav - emuinfo['gamma0']) / (emuinfo['gammaUB'] - emuinfo['gammaLB'])
    loglik = logs2sum + 0.5 * logdetRsum + 8 * np.sum(gammanorm ** 2)
    return loglik


def emulation_dlik_indp(gammav, emuinfo, fval=None):
    R, dR = emulation_getR(emuinfo, gammav, True)
    (cholR, pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        warnings.warn('Cholesky decomposition encounters a near-singular matrix', stacklevel=2)
        return np.inf

    mu = emulation_getmu(emuinfo, gammav)
    sigma2 = emulation_getsigma2(emuinfo, gammav)
    if fval is None:
        fval = emuinfo['f']
    resid = fval - mu
    missval = emuinfo['misval']

    dlogs2sum = np.zeros(gammav.shape)
    dlogdetsum = np.zeros(gammav.shape)
    donealr = np.zeros(emuinfo['m'])
    for k2 in range(0, emuinfo['m']):
        nhere = np.sum(1 - missval[:, k2]).astype(int)
        inds = (np.where(missval[:, k2] < 0.5)[0]).astype(int)
        Robs = R[inds, :][:, inds]
        (cholRobs, pd) = spla.lapack.dpotrf(Robs, True, True)
        if pd > 0.5:
            warnings.warn('Cholesky decomposition encounters a near-singular matrix', stacklevel=2)
            return np.inf
        Ri = spla.solve_triangular(cholRobs, spla.solve_triangular(cholRobs, np.diag(np.ones(nhere)), lower=True),
                                   lower=True, trans=True)
        addterm = np.zeros(gammav.shape)
        for l in range(0, emuinfo['hypstatparstructure'][0]):
            addterm[l] = np.sum(Ri * (dR[inds, :, l][:, inds]))

        for k in range(k2, emuinfo['m']):
            if (np.sum(np.abs(missval[:, k] ^ missval[:, k2])) < 0.5) and (donealr[k] < 0.5):
                dlogdetsum = dlogdetsum + addterm
                resvalhalf = spla.solve_triangular(cholRobs, resid[inds, k], lower=True)
                resvalinv = spla.solve_triangular(cholRobs, resvalhalf, lower=True, trans=True)
                s2calc = (1 / 2 * np.sum(resvalhalf * resvalhalf) + emuinfo['nu'] * sigma2[k])
                for l in range(0, emuinfo['hypstatparstructure'][0]):
                    dnum = - resvalinv.T @ dR[inds, :, l][:, inds] @ resvalinv
                    dlogs2sum[l] = dlogs2sum[l] + (1 / 2 * (nhere / 2 + emuinfo['nu']) * dnum) / s2calc
                for l in range(0, emuinfo['numcat']):
                    if (emuinfo['xcat'][k] == emuinfo['uniquecat'][l]):
                        dlogs2sum[emuinfo['hypstatparstructure'][1] + l] += emuinfo['nu'] * (
                                    ((nhere / 2 + emuinfo['nu']) * sigma2[k]) / s2calc - 1)
                        dnum = - 2 * np.sum(resvalinv)
                        dlogs2sum[emuinfo['hypstatparstructure'][2] + l] += (1 / 2 * (
                                    nhere / 2 + emuinfo['nu']) * dnum) / s2calc
                donealr[k] = 1
    dloglik = dlogs2sum + 0.5 * dlogdetsum + 16 * (gammav - emuinfo['gamma0']) / (
                (emuinfo['gammaUB'] - emuinfo['gammaLB']) ** 2)
    return dloglik


def emulation_lik_parasep(gammav, emuinfo, fval=None):
    R = emulation_getR(emuinfo, gammav, False)
    mu = emulation_getmu(emuinfo, gammav)
    sigma2 = emulation_getsigma2(emuinfo, gammav)
    if fval is None:
        fval = emuinfo['fpred']
    resid = fval - mu
    nhere = resid.shape[0]
    (cholR, pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        return np.inf
    logdetR = 2 * np.sum(np.log(np.diag(cholR)))
    residhalf = spla.solve_triangular(cholR, resid, lower=True)
    Sigmapart1 = emulation_getS(emuinfo, gammav, withdir=False)
    Sigma = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
    (cholSigma, pd) = spla.lapack.dpotrf(Sigma, True, True)

    invSigma = spla.solve_triangular(cholSigma,
                                     spla.solve_triangular(cholSigma, np.diag(np.ones(emuinfo['m'])), lower=True),
                                     lower=True, trans=True)

    if pd > 0.5:
        return np.inf
    else:
        logdetSigma = 2 * np.sum(np.log(np.diag(cholSigma)))
        gammanorm = (gammav - emuinfo['gamma0']) / (emuinfo['gammaUB'] - emuinfo['gammaLB'])
        loglik = np.sum(invSigma * (residhalf.T @ residhalf)) + nhere * logdetSigma + emuinfo[
            'm'] * logdetR + 8 * np.sum(gammanorm ** 2)
        return loglik


def emulation_dlik_parasep(gammav, emuinfo, fval=None):
    R, dR = emulation_getR(emuinfo, gammav, True)
    mu = emulation_getmu(emuinfo, gammav)
    sigma2 = emulation_getsigma2(emuinfo, gammav)
    if fval is None:
        fval = emuinfo['fpred']
    resid = fval - mu
    (cholR, pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        return np.inf
    nhere = resid.shape[0]

    invR = spla.solve_triangular(cholR, spla.solve_triangular(cholR, np.diag(np.ones(nhere)), lower=True), lower=True,
                                 trans=True)
    Sigmapart1, dSigmapart1 = emulation_getS(emuinfo, gammav, withdir=True)
    Sigma = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
    residhalf = spla.solve_triangular(cholR, resid, lower=True)
    residinv = spla.solve_triangular(cholR, residhalf, lower=True, trans=True)
    (cholSigma, pd) = spla.lapack.dpotrf(Sigma, True, True)
    if pd > 0.5:
        return np.inf
    invSigma = spla.solve_triangular(cholSigma,
                                     spla.solve_triangular(cholSigma, np.diag(np.ones(emuinfo['m'])), lower=True),
                                     lower=True, trans=True)

    if pd < 0.5:
        dlogdetR = np.zeros(gammav.shape)
        dlogdetSigma = np.zeros(gammav.shape)
        ddosomestuff = np.zeros(gammav.shape)
        for k in range(0, emuinfo['hypstatparstructure'][0]):
            A1 = invR @ np.squeeze(dR[:, :, k])
            A2 = residinv.T @ np.squeeze(dR[:, :, k]) @ residinv
            dlogdetR[k] = np.sum(invR * np.squeeze(dR[:, :, k]))
            ddosomestuff[k] = -np.sum(invSigma * A2)
        for k in range(emuinfo['hypstatparstructure'][0], emuinfo['hypstatparstructure'][1]):
            A3 = (np.diag(np.sqrt(sigma2)) @ np.squeeze(dSigmapart1[:, :, k - emuinfo['hypstatparstructure'][0]]) @ (
                np.diag(np.sqrt(sigma2))))
            dlogdetSigma[k] = np.sum(invSigma * A3)
            ddosomestuff[k] = -np.sum((invSigma @ A3 @ invSigma) * (residhalf.T @ residhalf))
        for k in range(emuinfo['hypstatparstructure'][1], emuinfo['hypstatparstructure'][2]):
            typevalnow = k - emuinfo['hypstatparstructure'][1]
            Dhere = np.zeros(emuinfo['m'])
            Dhere[emuinfo['xcat'] == emuinfo['uniquecat'][typevalnow]] = 1
            ddosomestuff[k + emuinfo['numcat']] = - 2 * np.sum((resid.T @ invR @ (0 * resid + Dhere)) * invSigma)
            A4 = 0.5 * (np.diag((Dhere / np.sqrt(sigma2))) @ Sigmapart1 @ np.diag((np.sqrt(sigma2))) + np.diag(
                (np.sqrt(sigma2))) @ Sigmapart1 @ np.diag((Dhere / np.sqrt(sigma2))))
            dlogdetSigma[k] = np.sum(invSigma * A4) * np.exp(gammav[k])
            ddosomestuff[k] = -np.sum((invSigma @ A4 @ invSigma) * (residhalf.T @ residhalf)) * np.exp(gammav[k])
        dloglik = nhere * dlogdetSigma + ddosomestuff + emuinfo['m'] * dlogdetR + 16 * (gammav - emuinfo['gamma0']) / (
                    (emuinfo['gammaUB'] - emuinfo['gammaLB']) ** 2)
        return dloglik
    else:
        return np.inf


def emulation_getR(emuinfo, gammav=None, withdir=False, diffTheta=None, sameTheta=None, thetasubset=None):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammathetacovhyp = gammav[0:emuinfo['hypstatparstructure'][0]]
    if thetasubset is None:
        return emuinfo['covthetaf'](emuinfo['theta'], emuinfo['theta'], gammathetacovhyp, returndir=withdir,
                                    diffX=diffTheta, sameX=sameTheta)
    else:
        return emuinfo['covthetaf'](emuinfo['theta'][thetasubset, :], emuinfo['theta'][thetasubset, :],
                                    gammathetacovhyp, returndir=withdir, diffX=diffTheta, sameX=sameTheta)


def emulation_getmu(emuinfo, gammav=None):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammamuhyp = gammav[(emuinfo['hypstatparstructure'][2]):]
    mu = np.ones(emuinfo['m'])
    for k in range(0, gammamuhyp.shape[0]):
        mu[emuinfo['xcat'] == emuinfo['uniquecat'][k]] = gammamuhyp[k]

    return mu


def emulation_getsigma2(emuinfo, gammav=None):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammasigmasqhyp = gammav[(emuinfo['hypstatparstructure'][1]):(emuinfo['hypstatparstructure'][2])]
    sigma2 = np.ones(emuinfo['m'])
    for k in range(0, gammasigmasqhyp.shape[0]):
        sigma2[emuinfo['xcat'] == emuinfo['uniquecat'][k]] = np.exp(gammasigmasqhyp[k])

    return sigma2


def emulation_getS(emuinfo, gammav=None, withdir=False):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammaxcovhyp = gammav[(emuinfo['hypstatparstructure'][0]):(emuinfo['hypstatparstructure'][1])]
    if withdir:
        return emuinfo['covxf'](emuinfo['xval'], emuinfo['xval'], gammaxcovhyp, type1=emuinfo['xcat'],
                                type2=emuinfo['xcat'], returndir=True)
    else:
        return emuinfo['covxf'](emuinfo['xval'], emuinfo['xval'], gammaxcovhyp, type1=emuinfo['xcat'],
                                type2=emuinfo['xcat'], returndir=False)


def emulation_matrixblockerfunction(emuinfo):
    missingmat = emuinfo['misval']
    Rs = np.zeros((np.int(np.sum(missingmat)), missingmat.shape[0]))
    Cs = np.zeros((np.int(np.sum(missingmat)), missingmat.shape[1]))

    ivaltr = np.where(np.sum(missingmat, 1))[0]
    jvaltr = np.where(np.sum(missingmat, 0))[0]
    missingmat = missingmat[np.ix_(ivaltr, jvaltr)]
    k = 0
    while (np.sum(missingmat) > 0.5):
        ival = np.where(np.sum(missingmat, 1))[0]
        jval = np.where(np.sum(missingmat, 0))[0]
        ivaltr = ivaltr[ival]
        jvaltr = jvaltr[jval]
        missingmat = missingmat[np.ix_(ival, jval)]
        n = missingmat.shape[0]
        # if emuinfo['blocking'] == 'individual':
        istar = 0
        jstar = np.where(missingmat[0, :])[0][0]
        Rs[k, ivaltr[istar]] = 1
        Cs[k, jvaltr[jstar]] = 1
        missingmat[istar, jstar] = 0
        if n == 1:
            k = k + 1
            break

        k = k + 1
    Rs = Rs[:k, ]
    Cs = Cs[:k, ]
    return Rs, Cs


def emulation_imputeiter(emuinfo, Rs, Cs, tol=10 ** (-6)):
    tolcutoff = tol * np.mean(np.abs(
        emuinfo['fpred'][np.where(1 - emuinfo['misval'])] - np.mean(emuinfo['fpred'][np.where(1 - emuinfo['misval'])])))

    for itera in range(0, 800):
        fpredbefore = 1 * emuinfo['fpred'][np.where(emuinfo['misval'])]
        Sinv = emuinfo['S_inv']
        for k in range(0, Rs.shape[0]):
            emulation_blockimputation(emuinfo, Rs[k, :], Cs[k, :], Sinv=Sinv)
        scalev = np.mean(np.abs(emuinfo['fpred'][np.where(emuinfo['misval'])] - fpredbefore))

        if scalev < tolcutoff:
            break
    return emuinfo


def emulation_blockimputation(emuinfo, paramiso, inputmiso, Sinv):
    paramis = np.array(np.where(paramiso)[0])
    inputmis = np.array(np.where(inputmiso)[0])
    inputobs = np.array(np.where(inputmiso < 0.5)[0])

    residnew = emulation_blockingsubfunc_emg(paramis, inputmis, inputobs, Sinv, emuinfo['R_inv'],
                                             emuinfo['fpred'] - emuinfo['mu'])
    emuinfo['fpred'] = residnew + emuinfo['mu']

    return emuinfo


def emulation_blockingsubfunc_emg(paramis, inputmis, inputobs, S_inv, R_inv, resid, doDraw=False):
    cholresidvarR = np.linalg.inv(spla.lapack.dpotrf(R_inv[paramis, :][:, paramis], True, True)[0])
    a3 = np.matmul(np.matmul(cholresidvarR, R_inv[paramis, :]), resid)

    a21 = np.matmul(cholresidvarR.T, a3)

    b9 = np.linalg.solve(S_inv[inputmis, :][:, inputmis], S_inv[inputmis, :])
    resid[np.ix_(paramis, inputmis)] -= np.matmul(a21, b9.T)

    if doDraw:
        S_inv_22_chol_draw, pr = spla.lapack.dpotrf(
            np.linalg.inv(0.5 * (S_inv[inputmis, :][:, inputmis] + (S_inv[inputmis, :][:, inputmis]).T)), True, True)
        if pr < 0.5:
            resid[np.ix_(paramis, inputmis)] += np.matmul(
                np.matmul(cholresidvarR, np.random.normal(0, 1, (cholresidvarR.shape[1], S_inv_22_chol_draw.shape[1]))),
                S_inv_22_chol_draw.T)

    return resid
