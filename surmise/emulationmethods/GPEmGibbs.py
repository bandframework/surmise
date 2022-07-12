""" """
import warnings
import numpy as np
import scipy.optimize as spo
import scipy.linalg as spla
from surmise.emulationsupport.GPEmGibbs_covmat_helper import setxcovf, setthetacovf

def fit(fitinfo, x, theta, f, misval=None, cat=False,
        xcovfname='matern', thetacovfname='matern', **kwargs):
    '''

    '''
    f = f.T

    fitinfo['theta'] = theta
    fitinfo['x'] = x
    fitinfo['f'] = f  # here
    fitinfo['cat'] = cat
    fitinfo['xcovfname'] = xcovfname
    fitinfo['thetacovfname'] = thetacovfname
    if misval is not None:
        if np.isnan(f).any() and (misval != np.isnan(f)).any():
            fitinfo['misval'] = misval ^ np.isnan(f)
            warnings.warn('''The provided missing value matrix (mis) 
            is updated to include NaN values.''')
    else:
        misval = np.isnan(f)

    emuinfo = __initialize(fitinfo, misval)

    emulation_hypest(emuinfo, 'indp')

    if emuinfo['misbool']:
        Rs,Cs = emulation_matrixblockerfunction(emuinfo)
        emulation_hypest(emuinfo)
        emulation_imputeiter(emuinfo, Rs, Cs, tol=10 ** (-3))
        emulation_hypest(emuinfo)
        emulation_imputeiter(emuinfo, Rs, Cs)
        emulation_hypest(emuinfo)
        emulation_imputeiter(emuinfo, Rs, Cs, tol=10 ** (-7))
        emulation_hypest(emuinfo)
    else:
        emulation_hypest(emuinfo)

    fitinfo['emuinfo'] = emuinfo
    return


def predict(predinfo, fitinfo, x, theta, computecov=True, **kwargs):
    emuinfo = fitinfo['emuinfo']

    r = emuinfo['covthetaf'](theta, emuinfo['theta'], emuinfo['gammathetacovhyp'])
    yhat = emuinfo['mu'] + r @ emuinfo['residinv']

    predinfo['mean'] = yhat
    return


def __initialize(fitinfo, misval):
    emuinfo = {}
    f = fitinfo['f']
    x = fitinfo['x']
    theta = fitinfo['theta']
    n, m = f.shape
    xval = (x[:, 0:(x.shape[1]-1)]).astype(float)
    if fitinfo['cat']:
        xcat = x[:, (x.shape[1]-1)]
    else:
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
    emuinfo['blocking'] = 'individual'
    emuinfo['modeltype'] = 'parasep'
    emuinfo['em_gibbs'] = True

    setxcovf(emuinfo, fitinfo['xcovfname'])
    setthetacovf(emuinfo, fitinfo['thetacovfname'])

    gammasigmasq0 = np.ones(uniquecat.shape[0])
    gammasigmasqLB = np.ones(uniquecat.shape[0])
    gammasigmasqUB = np.ones(uniquecat.shape[0])
    gammamu0 = np.ones(uniquecat.shape[0])
    gammamuLB = np.ones(uniquecat.shape[0])
    gammamuUB = np.ones(uniquecat.shape[0])
    for k in range(numcat):
        fconsid = f[np.squeeze(np.where(xcat == uniquecat[k]))]
        meanvalsnow = np.nanmean(fconsid, 1)
        gammasigmasq0[k] = 2+np.log(np.nanvar(fconsid))
        gammasigmasqLB[k] = 0 + gammasigmasq0[k]
        gammasigmasqUB[k] = 8 + gammasigmasq0[k]
        gammamu0[k] = np.nanmean(meanvalsnow)
        gammamuLB[k] = np.nanmin(meanvalsnow) - (np.nanmax(meanvalsnow)-np.nanmin(meanvalsnow))
        gammamuUB[k] = np.nanmax(meanvalsnow) + (np.nanmax(meanvalsnow)-np.nanmin(meanvalsnow))
    hypstatparstructure = [emuinfo['gammathetacovhyp0'].shape[0],
                           emuinfo['gammathetacovhyp0'].shape[0]+emuinfo['gammaxcovhyp0'].shape[0],
                           emuinfo['gammathetacovhyp0'].shape[0]+emuinfo['gammaxcovhyp0'].shape[0]+numcat,
                           emuinfo['gammathetacovhyp0'].shape[0]+emuinfo['gammaxcovhyp0'].shape[0]+numcat+numcat]
    emuinfo['gamma0'] = np.concatenate((emuinfo['gammathetacovhyp0'], emuinfo['gammaxcovhyp0'], gammasigmasq0, gammamu0))
    emuinfo['gammaLB'] = np.concatenate((emuinfo['gammathetacovhypLB'], emuinfo['gammaxcovhypLB'], gammasigmasqLB, gammamuLB))
    emuinfo['gammaUB'] = np.concatenate((emuinfo['gammathetacovhypUB'], emuinfo['gammaxcovhypUB'], gammasigmasqUB, gammamuUB))
    emuinfo['hypstatparstructure'] = hypstatparstructure
    emuinfo['gammahat'] = emuinfo['gamma0']

    return emuinfo


def predictmean(predinfo, **kwargs):
    return predinfo['mean']


def predictvar(predinfo, **kwargs):
    return predinfo['var']


def emulation_hypest(emuinfo, modeltype=None):
    if modeltype is None:
        modeltype = emuinfo['modeltype']

    if modeltype == 'nonparasep':
        try:
            likeattempt = emulation_lik_nonparasep(emuinfo['gammahat'], emuinfo)
            if likeattempt is float("inf"):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_nonparasep(emuinfo['gammahat'],emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt),1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_nonparasep(emuinfo['gammahat'],emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt),1))

        bounds = spo.Bounds(emuinfo['gammaLB'],emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_nonparasep, emuinfo['gammahat'],
                            args=(emuinfo),
                            method='L-BFGS-B',
                            options={'disp': False, 'ftol': myftol},
                            jac = emulation_dlik_nonparasep,
                            bounds=bounds)
        emuinfo['gammahat'] = opval.x
        emuinfo['R'] = emulation_getR(emuinfo)
        emuinfo['mu'] = emulation_getmu(emuinfo)

#        print(emuinfo['hypstatparstructure'])
        emuinfo['gammathetacovhyp'] = emuinfo['gammahat'][0:(emuinfo['hypstatparstructure'][0])]
        emuinfo['gammaxcovhyp'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][0]):(emuinfo['hypstatparstructure'][1])]
        emuinfo['gammasigmasq'] = np.exp(emuinfo['gammahat'][(emuinfo['hypstatparstructure'][1]):(emuinfo['hypstatparstructure'][2])])
        emuinfo['gammamu'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][2]):]

        R = emulation_getR(emuinfo)
        mu = emulation_getmu(emuinfo)
        sigma2 = emulation_getsigma2(emuinfo)

        Sigmapart1 = emulation_getS(emuinfo)
        emuinfo['S_prior'] = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
        emuinfo['Phi_prior'] = emuinfo['nu'] * emuinfo['S_prior']
        emuinfo['R_chol'] , _ = spla.lapack.dpotrf(emuinfo['R'],True,True)

        emuinfo['R_inv'] = spla.solve_triangular(emuinfo['R_chol'], spla.solve_triangular(emuinfo['R_chol'], np.diag(np.ones(emuinfo['n'])), lower=True), lower=True, trans=True)

        resid = emuinfo['fpred'] - emuinfo['mu']
        residhalf = spla.solve_triangular(emuinfo['R_chol'], resid, lower=True)
        emuinfo['residinv'] = spla.solve_triangular(emuinfo['R_chol'], residhalf, lower=True, trans=True)

        emuinfo['Phi_post'] = emuinfo['Phi_prior'] + np.matmul(residhalf.transpose(),residhalf)
        emuinfo['nu_post'] = emuinfo['n'] + emuinfo['nu']
        emuinfo['S_post'] = emuinfo['Phi_post'] / emuinfo['nu_post']


        emuinfo['Phi_post_chol'] , _ = spla.lapack.dpotrf(emuinfo['Phi_post'],True,True)
        emuinfo['S_post_chol'] = emuinfo['Phi_post_chol'] / np.sqrt(emuinfo['nu_post'])

        emuinfo['Phi_post_inv'] = spla.solve_triangular(emuinfo['Phi_post_chol'], spla.solve_triangular(emuinfo['Phi_post_chol'], np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)

        emuinfo['pw'] = np.linalg.solve(emuinfo['S_post_chol'].T, np.linalg.solve(emuinfo['S_post_chol'], emuinfo['residinv'].transpose()))
    if modeltype == 'parasep':
        try:
            likeattempt = emulation_lik_parasep(emuinfo['gammahat'],emuinfo)
            if likeattempt is float("inf"):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_parasep(emuinfo['gammahat'],emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt),1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_parasep(emuinfo['gammahat'],emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt),1))
        bounds = spo.Bounds(emuinfo['gammaLB'],emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_parasep, emuinfo['gammahat'],
                    args=(emuinfo),
                    method='L-BFGS-B',
                    options={'disp': False, 'ftol': myftol},
                    jac = emulation_dlik_parasep,
                    bounds=bounds)
        emuinfo['gammahat'] = opval.x
        emuinfo['R'] = emulation_getR(emuinfo)
        emuinfo['mu'] = emulation_getmu(emuinfo)

#        print(emuinfo['hypstatparstructure'])
        emuinfo['gammathetacovhyp'] =  emuinfo['gammahat'][0:(emuinfo['hypstatparstructure'][0])]
        emuinfo['gammaxcovhyp'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][0]):(emuinfo['hypstatparstructure'][1])]
        emuinfo['gammasigmasq'] = np.exp(emuinfo['gammahat'][(emuinfo['hypstatparstructure'][1]):(emuinfo['hypstatparstructure'][2])])
        emuinfo['gammamu'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][2]):]

        R = emulation_getR(emuinfo)
        mu = emulation_getmu(emuinfo)
        sigma2 = emulation_getsigma2(emuinfo)

        Sigmapart1 = emulation_getS(emuinfo)
        emuinfo['S'] = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
        emuinfo['R_chol'] , _ = spla.lapack.dpotrf(emuinfo['R'],True,True)

        emuinfo['R_inv'] = spla.solve_triangular(emuinfo['R_chol'], spla.solve_triangular(emuinfo['R_chol'], np.diag(np.ones(emuinfo['n'])), lower=True), lower=True, trans=True)

        resid = emuinfo['fpred'] - emuinfo['mu']
        residhalf = spla.solve_triangular(emuinfo['R_chol'], resid, lower=True)
        emuinfo['residinv'] = spla.solve_triangular(emuinfo['R_chol'], residhalf, lower=True, trans=True)

        emuinfo['S_chol'], _ = spla.lapack.dpotrf(emuinfo['S'],True,True)
        emuinfo['S_inv'] = spla.solve_triangular(emuinfo['S_chol'], spla.solve_triangular(emuinfo['S_chol'], np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)

        emuinfo['pw'] = np.linalg.solve(emuinfo['S_chol'].T, np.linalg.solve(emuinfo['S_chol'], emuinfo['residinv'].transpose()))
    else:
        #emulation_likderivativetester(emulation_lik_indp, emulation_dlik_indp, emuinfo)
        #asdas
        try:
            likeattempt = emulation_lik_indp(emuinfo['gammahat'],emuinfo)
            if likeattempt is float("inf"):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_indp(emuinfo['gammahat'],emuinfo)
            myftol = 0.25 / np.max((np.abs(likeattempt),1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_indp(emuinfo['gammahat'],emuinfo)
            myftol = 0.25 / np.max((np.abs(likeattempt),1))

        bounds = spo.Bounds(emuinfo['gammaLB'],emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_indp, emuinfo['gammahat'],
                            args=(emuinfo),
                            method='L-BFGS-B',
                            options={'disp': False, 'ftol': myftol},
                            jac = emulation_dlik_indp,
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
                inds = (np.where(emuinfo['misval'][:,k2] < 0.5)[0]).astype(int)
                ninds = (np.where(emuinfo['misval'][:,k2] > 0.5)[0]).astype(int)
                Robs = emuinfo['R'][np.ix_(inds,inds)]
                Rnobsobs = emuinfo['R'][np.ix_(ninds,inds)]
                cholRobs, _ = spla.lapack.dpotrf(Robs,True,True)
                for k in range(k2, emuinfo['m']):
                    if (np.sum(np.abs(emuinfo['misval'][:,k] ^ emuinfo['misval'][:,k2])) < 0.5) and (donealr[k] < 0.5):
                        resvalhalf = spla.solve_triangular(cholRobs, resid[inds,k], lower=True)
                        resvalinv = spla.solve_triangular(cholRobs, resvalhalf, lower=True, trans=True)
                        fpred[ninds,k] = emuinfo['mu'][k] + np.matmul(Rnobsobs,resvalinv)
                        fpred[inds,k] = emuinfo['f'][inds,k]
                        donealr[k] = 1
            emuinfo['fpred'] = fpred[:]
        else:
            emuinfo['fpred'] = emuinfo['f'][:]
    #print(np.max(np.abs(emuinfo['fpred'] - emuinfo['f'])))

    return None



def __hypest(emuinfo, modeltype = None):
    if modeltype is None:
        modeltype = emuinfo['modeltype']

    if modeltype == 'nonparasep':
        #dasd
        try:
            likeattempt = emulation_lik_nonparasep(emuinfo['gammahat'],emuinfo)
            if likeattempt is float("inf"):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_nonparasep(emuinfo['gammahat'],emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt),1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_nonparasep(emuinfo['gammahat'],emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt),1))

        bounds = spo.Bounds(emuinfo['gammaLB'],emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_nonparasep, emuinfo['gammahat'],
                            args=(emuinfo),
                            method='L-BFGS-B',
                            options={'disp': False, 'ftol': myftol},
                            jac = emulation_dlik_nonparasep,
                            bounds=bounds)
        emuinfo['gammahat'] = opval.x
        emuinfo['R'] = emulation_getR(emuinfo)
        emuinfo['mu'] = emulation_getmu(emuinfo)

#        print(emuinfo['hypstatparstructure'])
        emuinfo['gammathetacovhyp'] =  emuinfo['gammahat'][0:(emuinfo['hypstatparstructure'][0])]
        emuinfo['gammaxcovhyp'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][0]):(emuinfo['hypstatparstructure'][1])]
        emuinfo['gammasigmasq'] = np.exp(emuinfo['gammahat'][(emuinfo['hypstatparstructure'][1]):(emuinfo['hypstatparstructure'][2])])
        emuinfo['gammamu'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][2]):]

        R = emulation_getR(emuinfo)
        mu = emulation_getmu(emuinfo)
        sigma2 = emulation_getsigma2(emuinfo)

        Sigmapart1 = emulation_getS(emuinfo)
        emuinfo['S_prior'] = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
        emuinfo['Phi_prior'] = emuinfo['nu'] * emuinfo['S_prior']
        emuinfo['R_chol'] , _ = spla.lapack.dpotrf(emuinfo['R'],True,True)

        emuinfo['R_inv'] = spla.solve_triangular(emuinfo['R_chol'], spla.solve_triangular(emuinfo['R_chol'], np.diag(np.ones(emuinfo['n'])), lower=True), lower=True, trans=True)

        resid = emuinfo['fpred'] - emuinfo['mu']
        residhalf = spla.solve_triangular(emuinfo['R_chol'], resid, lower=True)
        emuinfo['residinv'] = spla.solve_triangular(emuinfo['R_chol'], residhalf, lower=True, trans=True)

        emuinfo['Phi_post'] = emuinfo['Phi_prior'] + np.matmul(residhalf.transpose(),residhalf)
        emuinfo['nu_post'] = emuinfo['n'] + emuinfo['nu']
        emuinfo['S_post'] = emuinfo['Phi_post'] / emuinfo['nu_post']


        emuinfo['Phi_post_chol'] , _ = spla.lapack.dpotrf(emuinfo['Phi_post'],True,True)
        emuinfo['S_post_chol'] = emuinfo['Phi_post_chol'] / np.sqrt(emuinfo['nu_post'])

        emuinfo['Phi_post_inv'] = spla.solve_triangular(emuinfo['Phi_post_chol'], spla.solve_triangular(emuinfo['Phi_post_chol'], np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)

        emuinfo['pw'] = np.linalg.solve(emuinfo['S_post_chol'].T, np.linalg.solve(emuinfo['S_post_chol'], emuinfo['residinv'].transpose()))
    if modeltype == 'parasep':
        try:
            likeattempt = emulation_lik_parasep(emuinfo['gammahat'],emuinfo)
            if likeattempt is float("inf"):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_parasep(emuinfo['gammahat'],emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt),1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_parasep(emuinfo['gammahat'],emuinfo)
            myftol = 0.1 / np.max((np.abs(likeattempt),1))
        bounds = spo.Bounds(emuinfo['gammaLB'],emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_parasep, emuinfo['gammahat'],
                    args=(emuinfo),
                    method='L-BFGS-B',
                    options={'disp': False, 'ftol': myftol},
                    jac = emulation_dlik_parasep,
                    bounds=bounds)
        emuinfo['gammahat'] = opval.x
        emuinfo['R'] = emulation_getR(emuinfo)
        emuinfo['mu'] = emulation_getmu(emuinfo)

#        print(emuinfo['hypstatparstructure'])
        emuinfo['gammathetacovhyp'] =  emuinfo['gammahat'][0:(emuinfo['hypstatparstructure'][0])]
        emuinfo['gammaxcovhyp'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][0]):(emuinfo['hypstatparstructure'][1])]
        emuinfo['gammasigmasq'] = np.exp(emuinfo['gammahat'][(emuinfo['hypstatparstructure'][1]):(emuinfo['hypstatparstructure'][2])])
        emuinfo['gammamu'] = emuinfo['gammahat'][(emuinfo['hypstatparstructure'][2]):]

        R = emulation_getR(emuinfo)
        mu = emulation_getmu(emuinfo)
        sigma2 = emulation_getsigma2(emuinfo)

        Sigmapart1 = emulation_getS(emuinfo)
        emuinfo['S'] = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
        emuinfo['R_chol'] , _ = spla.lapack.dpotrf(emuinfo['R'],True,True)

        emuinfo['R_inv'] = spla.solve_triangular(emuinfo['R_chol'], spla.solve_triangular(emuinfo['R_chol'], np.diag(np.ones(emuinfo['n'])), lower=True), lower=True, trans=True)

        resid = emuinfo['fpred'] - emuinfo['mu']
        residhalf = spla.solve_triangular(emuinfo['R_chol'], resid, lower=True)
        emuinfo['residinv'] = spla.solve_triangular(emuinfo['R_chol'], residhalf, lower=True, trans=True)

        emuinfo['S_chol'], _ = spla.lapack.dpotrf(emuinfo['S'],True,True)
        emuinfo['S_inv'] = spla.solve_triangular(emuinfo['S_chol'], spla.solve_triangular(emuinfo['S_chol'], np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)

        emuinfo['pw'] = np.linalg.solve(emuinfo['S_chol'].T, np.linalg.solve(emuinfo['S_chol'], emuinfo['residinv'].transpose()))
    else:
        #emulation_likderivativetester(emulation_lik_indp, emulation_dlik_indp, emuinfo)
        #asdas
        try:
            likeattempt = emulation_lik_indp(emuinfo['gammahat'],emuinfo)
            if likeattempt is float("inf"):
                emuinfo['gammahat'] = emuinfo['gamma0']
                likeattempt = emulation_lik_indp(emuinfo['gammahat'],emuinfo)
            myftol = 0.25 / np.max((np.abs(likeattempt),1))
        except:
            emuinfo['gammahat'] = emuinfo['gamma0']
            likeattempt = emulation_lik_indp(emuinfo['gammahat'],emuinfo)
            myftol = 0.25 / np.max((np.abs(likeattempt),1))

        bounds = spo.Bounds(emuinfo['gammaLB'],emuinfo['gammaUB'])
        opval = spo.minimize(emulation_lik_indp, emuinfo['gammahat'],
                            args=(emuinfo),
                            method='L-BFGS-B',
                            options={'disp': False, 'ftol': myftol},
                            jac = emulation_dlik_indp,
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
                inds = (np.where(emuinfo['misval'][:,k2] < 0.5)[0]).astype(int)
                ninds = (np.where(emuinfo['misval'][:,k2] > 0.5)[0]).astype(int)
                Robs = emuinfo['R'][np.ix_(inds,inds)]
                Rnobsobs = emuinfo['R'][np.ix_(ninds,inds)]
                cholRobs, _ = spla.lapack.dpotrf(Robs,True,True)
                for k in range(k2, emuinfo['m']):
                    if (np.sum(np.abs(emuinfo['misval'][:,k]-emuinfo['misval'][:,k2])) < 0.5) and (donealr[k] < 0.5):
                        resvalhalf = spla.solve_triangular(cholRobs, resid[inds,k], lower=True)
                        resvalinv = spla.solve_triangular(cholRobs, resvalhalf, lower=True, trans=True)
                        fpred[ninds,k] = emuinfo['mu'][k] + np.matmul(Rnobsobs,resvalinv)
                        fpred[inds,k] = emuinfo['f'][inds,k]
                        donealr[k] = 1
            emuinfo['fpred'] = fpred[:]
        else:
            emuinfo['fpred'] = emuinfo['f'][:]
    #print(np.max(np.abs(emuinfo['fpred'] - emuinfo['f'])))

    return



def emulation_lik_indp(gammav, emuinfo, fval=None):
    # if emuinfo['thetasubset'] is None:
    R = emulation_getR(emuinfo, gammav, False)
    # else:
    #     R = emulation_getR(emuinfo, gammav, False, thetasubset = emuinfo['thetasubset'])
    (cholR, pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        print('here ind')
        return float("inf")

    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        # if emuinfo['thetasubset'] is None:
        resid = emuinfo['f'] - mu
        missval = emuinfo['misval']
        # else:
        #     resid = emuinfo['f'][emuinfo['thetasubset'], :] - mu
        #     missval = emuinfo['misval'][emuinfo['thetasubset'],:]
    else:
        # if emuinfo['thetasubset'] is None:
        resid = fval - mu
        missval = emuinfo['misval']
        # else:
        #     resid = fval[emuinfo['thetasubset'], :] - mu
        #     missval = emuinfo['misval'][emuinfo['thetasubset'],:]


    logs2sum = 0
    logdetRsum = 0
    donealr = np.zeros(emuinfo['m'])
    for k2 in range(0, emuinfo['m']):
        nhere = np.sum(1-missval[:,k2])
        inds = (np.where(missval[:,k2] < 0.5)[0]).astype(int)
        Robs = R[np.ix_(inds,inds)]
        #!!!
        (cholRobs ,pd) = spla.lapack.dpotrf(Robs, True, True) # essentially cholesky
        if pd > 0.5:
            print('here ind')
            return float("inf")
        logdetval = 2*np.sum(np.log(np.diag(cholRobs)))
        for k in range(k2, emuinfo['m']):
            if (np.sum(np.abs(missval[:,k] ^ missval[:,k2])) < 0.5) and (donealr[k] < 0.5):
                resvalhalf = spla.solve_triangular(cholRobs, resid[inds, k], lower=True)
                logs2sum += (nhere/2+emuinfo['nu']) * np.log(1/2 * np.sum(resvalhalf * resvalhalf) + emuinfo['nu']*sigma2[k]) - emuinfo['nu']*np.log(emuinfo['nu']*sigma2[k])
                logdetRsum += logdetval
                donealr[k] = 1
    gammanorm = (gammav-emuinfo['gamma0'])/(emuinfo['gammaUB']-emuinfo['gammaLB'])
    loglik = logs2sum + 0.5 * logdetRsum + 8*np.sum(gammanorm ** 2)
    return loglik

def emulation_dlik_indp(gammav, emuinfo, fval = None):
    # if emuinfo['thetasubset'] is None:
    R, dR = emulation_getR(emuinfo, gammav, True)
    # else:
    #     R, dR = emulation_getR(emuinfo, gammav, True, thetasubset = emuinfo['thetasubset'])
    (cholR, pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        print('here ind')
        return float("inf")

    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        # if emuinfo['thetasubset'] is None:
        resid = emuinfo['f'] - mu
        missval = emuinfo['misval']
        # else:
        #     resid = emuinfo['f'][emuinfo['thetasubset'], :] - mu
        #     missval = emuinfo['misval'][emuinfo['thetasubset'],:]
    else:
        # if emuinfo['thetasubset'] is None:
        resid = fval - mu
        missval = emuinfo['misval']
        # else:
        #     resid = fval[emuinfo['thetasubset'], :] - mu
        #     missval = emuinfo['misval'][emuinfo['thetasubset'],:]


    dlogs2sum = np.zeros(gammav.shape)
    dlogdetsum = np.zeros(gammav.shape)
    donealr = np.zeros(emuinfo['m'])
    for k2 in range(0, emuinfo['m']):
        nhere = np.sum(1-missval[:,k2]).astype(int)
        inds = (np.where(missval[:,k2] < 0.5)[0]).astype(int)
        Robs = R[inds,:][:,inds]
        (cholRobs ,pd) = spla.lapack.dpotrf(Robs, True, True)
        if pd > 0.5:
            print('here ind')
            return float("inf")
        Ri = spla.solve_triangular(cholRobs,spla.solve_triangular(cholRobs,np.diag(np.ones(nhere)), lower=True), lower=True, trans=True)
        addterm = np.zeros(gammav.shape)
        for l in range(0,emuinfo['hypstatparstructure'][0]):
            addterm[l] = np.sum(Ri*(dR[inds,:,l][:,inds]))

        for k in range(k2, emuinfo['m']):
            if (np.sum(np.abs(missval[:,k] ^ missval[:,k2])) < 0.5) and (donealr[k] < 0.5):
                dlogdetsum = dlogdetsum + addterm
                resvalhalf = spla.solve_triangular(cholRobs, resid[inds,k], lower=True)
                resvalinv = spla.solve_triangular(cholRobs, resvalhalf, lower=True, trans=True)
                s2calc = (1/2 * np.sum(resvalhalf * resvalhalf) + emuinfo['nu'] * sigma2[k])
                for l in range(0,emuinfo['hypstatparstructure'][0]):
                    dnum = - resvalinv.T @ dR[inds,:,l][:,inds] @ resvalinv
                    dlogs2sum[l] =   dlogs2sum[l] +  (1/2* (nhere/2+emuinfo['nu'] ) * dnum) / s2calc
                for l in range(0,emuinfo['numcat']):
                     if (emuinfo['xcat'][k] == emuinfo['uniquecat'][l]):
                         dlogs2sum[emuinfo['hypstatparstructure'][1] + l] += emuinfo['nu'] * (((nhere/2+emuinfo['nu'] ) * sigma2[k]) / s2calc-1)
                         dnum = - 2 * np.sum(resvalinv)
                         dlogs2sum[emuinfo['hypstatparstructure'][2] + l] += (1/2* (nhere/2+emuinfo['nu'] ) * dnum) / s2calc
                donealr[k] = 1
    dloglik = dlogs2sum + 0.5 * dlogdetsum + 16 * (gammav-emuinfo['gamma0'])/((emuinfo['gammaUB']-emuinfo['gammaLB']) ** 2)
    return dloglik


def emulation_lik_nonparasep(gammav, emuinfo, fval = None):
    # if emuinfo['thetasubset'] is None:
    R = emulation_getR(emuinfo, gammav, False)
    # else:
    #     R = emulation_getR(emuinfo, gammav, False, thetasubset = emuinfo['thetasubset'])


    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        # if emuinfo['thetasubset'] is None:
        resid = emuinfo['fpred'] - mu
        # else:
        #     resid = emuinfo['fpred'][emuinfo['thetasubset'], :] - mu
    else:
        # if emuinfo['thetasubset'] is None:
        resid = fval - mu
        # else:
        #     resid = fval[emuinfo['thetasubset'], :] - mu

    (cholR ,pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        return float("inf")

    nhere = resid.shape[0]
    logdetR = 2*np.sum(np.log(np.diag(cholR)))
    residhalf = spla.solve_triangular(cholR, resid, lower=True)
    Sigmapart1 = emulation_getS(emuinfo, gammav, withdir = False)
    Sigma = emuinfo['nu'] * (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
    (cholSigma ,pd) = spla.lapack.dpotrf(Sigma, True, True)
    if pd > 0.5:
        return float("inf")
    logdetSigma = 2*np.sum(np.log(np.diag(cholSigma)))
    Sigmapost = residhalf.T @ residhalf + Sigma
    (cholSigmapost ,pd) = spla.lapack.dpotrf(Sigmapost, True, True)
    if pd < 0.5:
        logdetSigmapost = 2*np.sum(np.log(np.diag(cholSigmapost)))
        gammanorm = (gammav-emuinfo['gamma0'])/(emuinfo['gammaUB']-emuinfo['gammaLB'])
        loglik = (nhere+emuinfo['nu'])*logdetSigmapost - emuinfo['nu']*logdetSigma + emuinfo['m']*logdetR + 8*np.sum(gammanorm ** 2)
        return loglik
    else:
        return float("inf")

def emulation_dlik_nonparasep(gammav, emuinfo, fval = None):
    # if emuinfo['thetasubset'] is None:
    R, dR = emulation_getR(emuinfo, gammav, True)
    # else:
    #     R, dR = emulation_getR(emuinfo, gammav, True, thetasubset = emuinfo['thetasubset'])

    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        # if emuinfo['thetasubset'] is None:
        resid = emuinfo['fpred'] - mu
        # else:
        #     resid = emuinfo['fpred'][emuinfo['thetasubset'], :] - mu
    else:
        # if emuinfo['thetasubset'] is None:
        resid = fval - mu
        # else:
        #     resid = fval[emuinfo['thetasubset'], :] - mu

    (cholR ,pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        return float("inf")

    nhere = resid.shape[0]
    invR = spla.solve_triangular(cholR,spla.solve_triangular(cholR,np.diag(np.ones(resid.shape[0])), lower=True), lower=True, trans=True)
    Sigmapart1, dSigmapart1 = emulation_getS(emuinfo, gammav, withdir = True)
    Sigma = emuinfo['nu'] * (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
    residhalf = spla.solve_triangular(cholR, resid, lower=True)
    residinv = spla.solve_triangular(cholR, residhalf, lower=True, trans=True)
    (cholSigma ,pd) = spla.lapack.dpotrf(Sigma, True, True)
    if pd > 0.5:
        return float("inf")
    invSigma = spla.solve_triangular(cholSigma,spla.solve_triangular(cholSigma,np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)
    Sigmapost = residhalf.T @ residhalf + Sigma
    (cholSigmapost ,pd) = spla.lapack.dpotrf(Sigmapost, True, True)
    if pd < 0.5:
        invSigmapost = spla.solve_triangular(cholSigmapost,spla.solve_triangular(cholSigmapost,np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)
        dlogdetR = np.zeros(gammav.shape)
        dlogdetSigma = np.zeros(gammav.shape)
        dlogdetSigmapost = np.zeros(gammav.shape)
        A5 = residinv @ invSigmapost @ residinv.T
        A12 = np.diag(np.sqrt(sigma2)) @ invSigmapost @ np.diag(np.sqrt(sigma2))
        A11 = np.diag(np.sqrt(sigma2)) @ invSigma @ np.diag(np.sqrt(sigma2))
        for k in range(0,emuinfo['hypstatparstructure'][0]):
            dlogdetR[k] = np.sum(invR * np.squeeze(dR[:,:,k]))
            dlogdetSigmapost[k] = -np.sum(A5 * np.squeeze(dR[:,:,k]))
        for k in range(emuinfo['hypstatparstructure'][0],emuinfo['hypstatparstructure'][1]):
            A10 = np.squeeze(dSigmapart1[:,:, k- emuinfo['hypstatparstructure'][0]])
            dlogdetSigmapost[k] = emuinfo['nu'] *np.sum(A12 * A10)
            dlogdetSigma[k] = emuinfo['nu'] *np.sum(A11 * A10)
        for k in range(emuinfo['hypstatparstructure'][1],emuinfo['hypstatparstructure'][2]):
            typevalnow = k-emuinfo['hypstatparstructure'][1]
            Dhere = np.zeros(emuinfo['m'])
            Dhere[emuinfo['xcat'] == emuinfo['uniquecat'][typevalnow]] = 1
            A3 = emuinfo['nu']/2*(np.diag((Dhere/np.sqrt(sigma2))) @ Sigmapart1 @ np.diag((np.sqrt(sigma2))) + np.diag((np.sqrt(sigma2))) @ Sigmapart1 @ np.diag((Dhere/np.sqrt(sigma2))))
            dlogdetSigmapost[k] = np.sum(invSigmapost*A3) * np.exp(gammav[emuinfo['hypstatparstructure'][1]+typevalnow])
            dlogdetSigma[k] = np.sum(invSigma*A3) * np.exp(gammav[emuinfo['hypstatparstructure'][1]+typevalnow])
            dresidval = spla.solve_triangular(cholR, np.squeeze(0*resid + Dhere), lower=True)
            A4 = dresidval.T @ residhalf + residhalf.T @ dresidval
            dlogdetSigmapost[k +emuinfo['hypstatparstructure'][2] - emuinfo['hypstatparstructure'][1]] = -np.sum(invSigmapost * A4)
        dloglik = (nhere+emuinfo['nu'])*dlogdetSigmapost - emuinfo['nu']*dlogdetSigma + emuinfo['m']*dlogdetR +  16 * (gammav-emuinfo['gamma0'])/((emuinfo['gammaUB']-emuinfo['gammaLB']) ** 2)
        return dloglik
    else:
        return float("inf")


def emulation_lik_parasep(gammav, emuinfo, fval = None):
    # if emuinfo['thetasubset'] is None:
    R = emulation_getR(emuinfo, gammav, False)
    # else:
    #     R = emulation_getR(emuinfo, gammav, False, thetasubset = emuinfo['thetasubset'])

    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        # if emuinfo['thetasubset'] is None:
            resid = emuinfo['fpred'] - mu
        # else:
        #     resid = emuinfo['fpred'][emuinfo['thetasubset'], :] - mu
    else:
        # if emuinfo['thetasubset'] is None:
        resid = fval - mu
        # else:
        #     resid = fval[emuinfo['thetasubset'], :] - mu
    nhere = resid.shape[0]
    (cholR ,pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        return float("inf")
    logdetR = 2*np.sum(np.log(np.diag(cholR)))
    residhalf = spla.solve_triangular(cholR, resid, lower=True)
    residinv = spla.solve_triangular(cholR, residhalf, lower=True, trans=True)
    Sigmapart1 = emulation_getS(emuinfo, gammav, withdir = False)
    Sigma = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
    (cholSigma ,pd) = spla.lapack.dpotrf(Sigma, True, True)


    invSigma = spla.solve_triangular(cholSigma,spla.solve_triangular(cholSigma,np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)

    if pd > 0.5:
        return float("inf")
    logdetSigma = 2*np.sum(np.log(np.diag(cholSigma)))
    if pd < 0.5:
        gammanorm = (gammav-emuinfo['gamma0'])/(emuinfo['gammaUB']-emuinfo['gammaLB'])
        dosomestuff = spla.solve_triangular(cholSigma, residhalf.T)
        loglik =np.sum(invSigma * (residhalf.T @ residhalf)) + nhere*logdetSigma + emuinfo['m']*logdetR + 8*np.sum(gammanorm ** 2)
        return loglik
    else:
        return float("inf")

def emulation_dlik_parasep(gammav, emuinfo, fval = None):
    # if emuinfo['thetasubset'] is None:
    R, dR = emulation_getR(emuinfo, gammav, True)
    # else:
    #     R, dR = emulation_getR(emuinfo, gammav, True, thetasubset = emuinfo['thetasubset'])

    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        # if emuinfo['thetasubset'] is None:
        resid = emuinfo['fpred'] - mu
        # else:
        #     resid = emuinfo['fpred'][emuinfo['thetasubset'], :] - mu
    else:
        # if emuinfo['thetasubset'] is None:
        resid = fval - mu
        # else:
        #     resid = fval[emuinfo['thetasubset'], :] - mu
    (cholR ,pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        return float("inf")
    nhere = resid.shape[0]

    invR = spla.solve_triangular(cholR,spla.solve_triangular(cholR,np.diag(np.ones(nhere)), lower=True), lower=True, trans=True)
    Sigmapart1, dSigmapart1 = emulation_getS(emuinfo, gammav, withdir = True)
    Sigma = (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
    residhalf = spla.solve_triangular(cholR, resid, lower=True)
    residinv = spla.solve_triangular(cholR, residhalf, lower=True, trans=True)
    (cholSigma ,pd) = spla.lapack.dpotrf(Sigma, True, True)
    if pd > 0.5:
        return float("inf")
    invSigma = spla.solve_triangular(cholSigma,spla.solve_triangular(cholSigma,np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)

    residinvinv = (invSigma @ residinv.T).T
    if pd < 0.5:
        dlogdetR = np.zeros(gammav.shape)
        dlogdetSigma = np.zeros(gammav.shape)
        ddosomestuff = np.zeros(gammav.shape)
        for k in range(0,emuinfo['hypstatparstructure'][0]):
            A1 = invR @ np.squeeze(dR[:,:,k])
            A2 = residinv.T @ np.squeeze(dR[:,:,k]) @ residinv
            dlogdetR[k] = np.sum(invR * np.squeeze(dR[:,:,k]))
            ddosomestuff[k] = -np.sum(invSigma * A2)
        for k in range(emuinfo['hypstatparstructure'][0],emuinfo['hypstatparstructure'][1]):
            A3 = (np.diag(np.sqrt(sigma2)) @ np.squeeze(dSigmapart1[:,:,k-emuinfo['hypstatparstructure'][0]]) @ (np.diag(np.sqrt(sigma2))))
            dlogdetSigma[k] = np.sum(invSigma*A3)
            ddosomestuff[k] = -np.sum((invSigma @ A3 @ invSigma) * (residhalf.T @ residhalf))
        for k in range(emuinfo['hypstatparstructure'][1],emuinfo['hypstatparstructure'][2]):
            typevalnow = k-emuinfo['hypstatparstructure'][1]
            Dhere = np.zeros(emuinfo['m'])
            Dhere[emuinfo['xcat'] == emuinfo['uniquecat'][typevalnow]] = 1
            ddosomestuff[k+emuinfo['numcat']] = - 2 * np.sum((resid.T @ invR @ (0*resid + Dhere)) * invSigma)
            A4 = 0.5*(np.diag((Dhere/np.sqrt(sigma2))) @ Sigmapart1 @ np.diag((np.sqrt(sigma2))) + np.diag((np.sqrt(sigma2))) @ Sigmapart1 @ np.diag((Dhere/np.sqrt(sigma2))))
            dlogdetSigma[k] = np.sum(invSigma*A4) * np.exp(gammav[k])
            ddosomestuff[k] = -np.sum((invSigma @ A4 @ invSigma) * (residhalf.T @ residhalf)) * np.exp(gammav[k])
        dloglik = nhere*dlogdetSigma + ddosomestuff + emuinfo['m']*dlogdetR +  16 * (gammav-emuinfo['gamma0'])/((emuinfo['gammaUB']-emuinfo['gammaLB']) ** 2)
        return dloglik
    else:
        return float("inf")


def emulation_getR(emuinfo, gammav = None, withdir = False, diffTheta = None, sameTheta = None, thetasubset = None):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammathetacovhyp = gammav[0:emuinfo['hypstatparstructure'][0]]
    if thetasubset is None:
        return emuinfo['covthetaf'](emuinfo['theta'], emuinfo['theta'], gammathetacovhyp, returndir = withdir, diffX = diffTheta, sameX = sameTheta)
    else:
        return emuinfo['covthetaf'](emuinfo['theta'][thetasubset,:], emuinfo['theta'][thetasubset,:], gammathetacovhyp, returndir = withdir, diffX = diffTheta, sameX = sameTheta)

def emulation_getmu(emuinfo, gammav = None, withdir = False):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammamuhyp = gammav[(emuinfo['hypstatparstructure'][2]):]
    mu = np.ones(emuinfo['m'])
    for k in range(0,gammamuhyp.shape[0]):
        mu[emuinfo['xcat'] == emuinfo['uniquecat'][k]] = gammamuhyp[k]

    return mu

def emulation_getsigma2(emuinfo, gammav = None, withdir = False):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammasigmasqhyp = gammav[(emuinfo['hypstatparstructure'][1]):(emuinfo['hypstatparstructure'][2])]
    sigma2 = np.ones(emuinfo['m'])
    for k in range(0,gammasigmasqhyp.shape[0]):
        sigma2[emuinfo['xcat'] == emuinfo['uniquecat'][k]] = np.exp(gammasigmasqhyp[k])

    return sigma2

def emulation_getS(emuinfo, gammav = None, withdir = False):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammaxcovhyp = gammav[(emuinfo['hypstatparstructure'][0]):(emuinfo['hypstatparstructure'][1])]
    if withdir:
        return emuinfo['covxf'](emuinfo['xval'], emuinfo['xval'], gammaxcovhyp ,type1 = emuinfo['xcat'], type2 = emuinfo['xcat'], returndir = True)
    else:
        return emuinfo['covxf'](emuinfo['xval'], emuinfo['xval'], gammaxcovhyp ,type1 = emuinfo['xcat'], type2 = emuinfo['xcat'], returndir = False)


def emulation_matrixblockerfunction(emuinfo):
    missingmat = emuinfo['misval']
    Rs = np.zeros((np.int(np.sum(missingmat)), missingmat.shape[0]))
    Cs = np.zeros((np.int(np.sum(missingmat)), missingmat.shape[1]))
    # breakintoblocks
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
        if emuinfo['blocking'] == 'individual':
            istar = 0
            jstar = np.where(missingmat[0, :])[0][0]
            Rs[k, ivaltr[istar]] = 1
            Cs[k, jvaltr[jstar]] = 1
            missingmat[istar, jstar] = 0
            if n == 1:
                k = k + 1
                break

        if emuinfo['blocking'] == 'row':
            if n == 1:
                Rs[k, ivaltr] = 1
                Cs[k, jvaltr] = 1
                k = k + 1
                break
            blockrow = np.zeros(missingmat.shape[0])
            blockrow[0] = 1
            blockcol = missingmat[0, :]
            Rs[k, ivaltr[np.where(blockrow)[0]]] = 1
            Cs[k, jvaltr[np.where(blockcol)[0]]] = 1
            missingmat[np.ix_(np.where(blockrow)[0], np.where(blockcol)[0])] = 0
        if emuinfo['blocking'] == 'column':
            if n == 1:
                Rs[k, ivaltr] = 1
                Cs[k, jvaltr] = 1
                k = k + 1
                break
            blockcol = np.zeros(missingmat.shape[1])
            blockcol[0] = 1
            blockrow = missingmat[:, 0]
            Rs[k, ivaltr[np.where(blockrow)[0]]] = 1
            Cs[k, jvaltr[np.where(blockcol)[0]]] = 1
            missingmat[np.ix_(np.where(blockrow)[0], np.where(blockcol)[0])] = 0
        else:
            if n == 1:
                Rs[k, ivaltr] = 1
                Cs[k, jvaltr] = 1
                k = k + 1
                break
            blockrow = np.zeros(missingmat.shape[0])
            blockrow[0] = 1
            blockcol = missingmat[0, :]

            numrow = np.sum(blockrow)
            numcol = np.sum(blockcol)

            newsize = np.zeros(n)
            for j in range(0, n):
                newsize = 0 * newsize
                for i in range(0, n):
                    if (blockrow[i] == 0):
                        numcolpot = np.sum(blockcol * missingmat[i, :])
                        newsize[i] = numcolpot * (numrow + 1)
                    else:
                        newsize[i] = 0
                if ((numrow * numcol + 0.5) < np.max(newsize)):
                    istar = np.argmax(newsize)
                    blockrow[istar] = 1
                    blockcol = blockcol * missingmat[istar, :]
                    numrow = numrow + 1
                    numcol = np.sum(blockcol)
                else:
                    break
            Rs[k, ivaltr[np.where(blockrow)[0]]] = 1
            Cs[k, jvaltr[np.where(blockcol)[0]]] = 1
            missingmat[np.ix_(np.where(blockrow)[0], np.where(blockcol)[0])] = 0
        k = k + 1
    Rs = Rs[:k, ]
    Cs = Cs[:k, ]
    return Rs, Cs



def emulation_imputeiter(emuinfo, Rs, Cs, tol = 10 ** (-6)):
    tolcutoff = tol*np.mean(np.abs(emuinfo['fpred'][np.where(1-emuinfo['misval'])]-np.mean(emuinfo['fpred'][np.where(1-emuinfo['misval'])])))

    for itera in range(0,800):
        fpredbefore = 1*emuinfo['fpred'][np.where(emuinfo['misval'])]
        if emuinfo['modeltype'] == 'parasep':
            Sinv = emuinfo['S_inv']
        else:
            if emuinfo['em_gibbs']:
                emuinfo['resid'] = emuinfo['fpred'] - emuinfo['mu']
                residhalf = spla.solve_triangular(emuinfo['R_chol'], emuinfo['resid'], lower=True)
                emuinfo['Phi_post'] = emuinfo['Phi_prior'] + np.matmul(residhalf.transpose(),residhalf)
                emuinfo['Phi_post_inv'] = np.linalg.inv(emuinfo['Phi_post'])
                Sinv = emuinfo['Phi_post_inv'] * (emuinfo['nu_post'] + emuinfo['m']+1)
            else:
                emuinfo['resid'] = emuinfo['fpred'] - emuinfo['mu']
                residhalf = spla.solve_triangular(emuinfo['R_chol'], emuinfo['resid'], lower=True)
                emuinfo['Phi_post'] = emuinfo['Phi_prior'] + np.matmul(residhalf.transpose(),residhalf)
                emuinfo['Phi_post_inv'] = np.linalg.inv(emuinfo['Phi_post'])
        for k in range(0,Rs.shape[0]):
            if emuinfo['em_gibbs'] or (emuinfo['modeltype'] == 'parasep'):
                emulation_blockimputation(emuinfo, Rs[k,:], Cs[k,:], Sinv = Sinv)
            else:
                try:
                    emulation_blockimputation(emuinfo, Rs[k,:], Cs[k,:])
                except:
                    emuinfo['resid'] = emuinfo['fpred'] - emuinfo['mu']
                    residhalf = spla.solve_triangular(emuinfo['R_chol'], emuinfo['resid'], lower=True)
                    emuinfo['Phi_post'] = emuinfo['Phi_prior'] + np.matmul(residhalf.transpose(),residhalf)
                    emuinfo['Phi_post_inv'] = np.linalg.inv(emuinfo['Phi_post'])
                    emulation_blockimputation(emuinfo, Rs[k,:], Cs[k,:])
        scalev = np.mean(np.abs(emuinfo['fpred'][np.where(emuinfo['misval'])]-fpredbefore))
        if (scalev < tolcutoff):
            break
    return emuinfo


def emulation_blockimputation(emuinfo, paramiso, inputmiso, Sinv = None):
    paramis = np.array(np.where(paramiso)[0])
    inputmis = np.array(np.where(inputmiso)[0])
    inputobs = np.array(np.where(inputmiso<0.5)[0])

    if Sinv is None:
        residnew, Phi, Phi_inv = emulation_blockingsubfunc(paramis, inputmis, inputobs, emuinfo['Phi_post'], emuinfo['Phi_post_inv'], emuinfo['R_inv'], emuinfo['fpred'] - emuinfo['mu'], emuinfo['nu_post'])
        emuinfo['Phi_post'] = Phi
        emuinfo['Phi_post_inv'] = Phi_inv
    else:
        residnew = emulation_blockingsubfunc_emg(paramis, inputmis, inputobs, Sinv, emuinfo['R_inv'], emuinfo['fpred'] - emuinfo['mu'])
    emuinfo['fpred'] = residnew + emuinfo['mu']

    return emuinfo


def emulation_blockingsubfunc_emg(paramis, inputmis, inputobs, S_inv, R_inv, resid, doDraw=False):
    cholresidvarR = np.linalg.inv(spla.lapack.dpotrf(R_inv[paramis, :][:, paramis], True, True)[0])
    a3 = np.matmul(np.matmul(cholresidvarR, R_inv[paramis, :]), resid)

    a21 = np.matmul(cholresidvarR.T, a3)

    b9 = np.linalg.solve(S_inv[inputmis, :][:, inputmis], S_inv[inputmis, :])
    resid[np.ix_(paramis, inputmis)] -= np.matmul(a21, b9.transpose())

    if doDraw:
        S_inv_22_chol_draw, pr = spla.lapack.dpotrf(
            np.linalg.inv(0.5 * (S_inv[inputmis, :][:, inputmis] + (S_inv[inputmis, :][:, inputmis]).T)), True, True)
        if (pr < 0.5):
            resid[np.ix_(paramis, inputmis)] += np.matmul(
                np.matmul(cholresidvarR, np.random.normal(0, 1, (cholresidvarR.shape[1], S_inv_22_chol_draw.shape[1]))),
                S_inv_22_chol_draw.transpose())

    return resid


def emulation_blockingsubfunc(paramis, inputmis, inputobs, Phi, Phi_inv, R_inv, resid, nu, doDraw=False):
    cholresidvarR = np.linalg.inv(spla.lapack.dpotrf(R_inv[paramis, :][:, paramis], True, True)[0])
    a3 = np.matmul(np.matmul(cholresidvarR, R_inv[paramis, :]), resid)

    a21 = np.matmul(cholresidvarR.T, a3)
    Phi_update = Phi - np.matmul(a3.transpose(), a3)
    if (a3.shape[1] > (1.5 * (a3.shape[0] - 1))):
        a4 = np.matmul(Phi_inv, a3.transpose())
        a8 = np.matmul(a4, np.linalg.solve(np.identity(a4.shape[1]) - np.matmul(a3, a4), a4.transpose()))
        Phi_update_inv = Phi_inv + 0.5 * (a8 + a8.T)
    else:
        Phi_update_inv = np.linalg.inv(Phi_update)

    b9 = np.linalg.solve(Phi_update_inv[inputmis, :][:, inputmis], Phi_update_inv[inputmis, :])
    resid[np.ix_(paramis, inputmis)] -= np.matmul(a21, b9.transpose())
    if doDraw:
        choltildePhi22now, _ = spla.lapack.dpotrf(Phi_update_inv[inputmis, :][:, inputmis], True, True)
        if (a3.shape[1] < (1.5 * (a3.shape[0] - 1))):
            Phi_update_11_inv = np.linalg.inv(Phi_update[inputobs, :][:, inputobs])
        else:
            Phi_update_11_inv = Phi_update_inv[inputobs, :][:, inputobs] - np.matmul(
                Phi_update_inv[inputobs, :][:, inputmis], b9[:, inputobs])
        Xforsamp = np.matmul(choltildePhi22now,
                             np.random.normal(0, 1, (choltildePhi22now.shape[1], nu - paramis.shape[0])))
        Phi_update_22_chol_draw, _ = spla.lapack.dpotrf(np.linalg.inv(np.matmul(Xforsamp, Xforsamp.T)), True, True)

        b5 = np.matmul(np.matmul(a21[:, inputobs], Phi_update_11_inv), a21[:, inputobs].T)
        mat2 = np.matmul(cholresidvarR,
                         spla.lapack.dpotrf(np.identity(a21.shape[0]) + 0.5 * (b5 + b5.T), True, True)[0])
        resid[np.ix_(paramis, inputmis)] += np.matmul(
            np.matmul(mat2, np.random.normal(0, 1, (mat2.shape[1], Phi_update_22_chol_draw.shape[1]))),
            Phi_update_22_chol_draw.transpose())

    a6 = np.matmul(np.matmul(cholresidvarR, R_inv[paramis, :]), resid)
    Phi = Phi_update + np.matmul(a6.transpose(), a6)
    if (a6.shape[1] > (1.5 * (a6.shape[0] - 1))):
        a10 = np.matmul(Phi_update_inv, a6.transpose())
        a12 = np.matmul(a10, np.linalg.solve(np.identity(a4.shape[1]) + np.matmul(a6, a10), a10.transpose()))
        Phi_inv = Phi_update_inv - 0.5 * (a12 + a12.T)
    else:
        Phi_inv = np.linalg.inv(Phi)

    return resid, Phi, Phi_inv
