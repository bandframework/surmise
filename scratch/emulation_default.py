# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:12:24 2020

@author: Matt
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:40:43 2020

@author: Matt
"""

import numpy as np
import scipy.linalg as spla
import scipy.optimize as spo
from emulation_default_covfuncs import *
from emulation_default_likelihoods import *

def emulation_builder_default(thetaval, fval, inputval, missingval, emuoptions = None):
    emuinfo = emulation_initialize(thetaval, fval, inputval, missingval, emuoptions)

    emulation_hypest(emuinfo, 'indp')
    if emuinfo['ismissingval']:
        Rs,Cs = emulation_matrixblockerfunction(emuinfo)
        emulation_hypest(emuinfo)
        emulation_imputeiter(emuinfo, Rs, Cs, tol = 10 ** (-3))
        emulation_hypest(emuinfo)
        emulation_imputeiter(emuinfo, Rs, Cs)
        emulation_hypest(emuinfo)
        emulation_imputeiter(emuinfo, Rs, Cs, tol = 10 ** (-7))
        emulation_hypest(emuinfo)
    else:
        emulation_hypest(emuinfo)

    if emuinfo['ismissingval']:
        emulation_sampliter(emuinfo, Rs, Cs)
    else:
        emulation_sampliter(emuinfo, None, None)

    return emuinfo

def emulation_prediction_default(emuinfo, thetanew):

    r = emuinfo['covthetaf'](thetanew, emuinfo['thetaval'], emuinfo['gammathetacovhyp'])
    yhat = emuinfo['mu'] + np.matmul(r, emuinfo['residinv'])

    predinfo = {
        'mean': yhat
    }

    return predinfo

def emulation_draws_default(emuinfo, thetanew, drawoptions):
    r = emuinfo['covthetaf'](thetanew, emuinfo['thetaval'], emuinfo['gammathetacovhyp'])

    if drawoptions is None:
        rowmarginals = True
        numsamples = 100
    else:
        if 'rowmarginals' not in drawoptions:
            rowmarginals = True
        else:
            rowmarginals = drawoptions['rowmarginals']
        if 'numsamples' not in drawoptions:
            numsamples = 100
        else:
            numsamples = drawoptions['numsamples']

    rnew = emuinfo['covthetaf'](thetanew, thetanew, emuinfo['gammathetacovhyp'])
    rcalc = spla.solve_triangular(emuinfo['R_chol'], r.transpose(), lower=True)
    residvar = rnew - np.matmul(rcalc.transpose(),rcalc)
    if rowmarginals:
        residvarchol = np.diag(np.sqrt(np.abs(np.diag(residvar))))
    else:
        (residvarchol ,pd) = spla.lapack.dpotrf(residvar, True, True)
        if pd > 0.5:
            print('numerical error. returning row marginals.  try again with fewer thetas')
            residvarv = np.sqrt(np.abs(np.diag(residvar)))

    fdraws = np.ones((thetanew.shape[0], emuinfo['m'], numsamples))
    for l2 in range(0,numsamples):
        l1 = l2 % emuinfo['pwsample'].shape[2]
        yhat = emuinfo['mu'] + np.matmul(r, emuinfo['pwsample'][:,:,l1])
        Schol = np.squeeze(emuinfo['Ssamplechol'][:,:,l1])
        if rowmarginals:
            residmis = np.matmul(np.matmul(residvarchol,np.random.normal(0,1,yhat.shape)),Schol.transpose())
        else:
            residmis = residvarv * (np.matmul(np.random.normal(0,1,yhat.shape),Schol.transpose()))

        fdraws[:,:,l2] = yhat + residmis

    return fdraws

def emulation_initialize(thetaval, fval, inputval, missingval, emuoptions):
    emuinfo = {}
    if missingval is not None:
        if (np.sum(missingval) < 0.5):
            emuinfo['ismissingval'] = False
            emuinfo['missingval'] = np.zeros(fval.shape)
        else:
            emuinfo['ismissingval'] = True
            emuinfo['missingval'] = missingval

    if missingval is None:
        emuinfo['ismissingval'] = False
        emuinfo['missingval'] = np.zeros(fval.shape)

    if emuoptions is None:
        emuinfo['blocking'] = 'optimal'
        emuinfo['em_gibbs'] = False
        emuinfo['ndrawsamples'] = None
        emuinfo['modeltype'] = 'nonparasep'
        emuinfo['corrxfname'] = 'matern'
        emuinfo['corrthetafname'] = 'matern'
    else:
        if 'blocking' not in emuoptions:
            emuinfo['blocking'] = 'optimal'
        else:
            emuinfo['blocking'] = emuoptions['blocking']
        if 'modeltype' not in emuoptions:
            emuinfo['modeltype'] = 'nonparasep'
        else:
            emuinfo['modeltype'] = emuoptions['modeltype']
        if 'corrthetafname' not in emuoptions:
            emuinfo['corrthetafname'] = 'matern'
        else:
            emuinfo['corrthetafname'] = emuoptions['corrthetafname']
        if 'corrxfname' not in emuoptions:
            emuinfo['corrxfname'] = 'matern'
        else:
            emuinfo['corrxfname'] = emuoptions['corrxfname']
        if 'em_gibbs' not in emuoptions:
            emuinfo['em_gibbs'] = False
        else:
            emuinfo['em_gibbs'] = emuoptions['em_gibbs']
        if 'ndrawsamples' not in emuoptions:
            emuinfo['ndrawsamples'] = None
        else:
            emuinfo['ndrawsamples'] = emuoptions['ndrawsamples']
    emuinfo['nu'] = np.max((fval.shape[1]+60, 3*fval.shape[1], 0.8*fval.shape[0])).astype('int')
    xval = (inputval[:,0:(inputval.shape[1]-1)]).astype('float')
    typeval = inputval[:,(inputval.shape[1]-1)]
    emuinfo['thetaval'] = thetaval
    emuinfo['fval'] = fval
    emuinfo['typeval'] = typeval
    emuinfo['xval'] = xval
    uniquetype = np.unique(typeval)
    numtype = uniquetype.shape[0]
    emuinfo['uniquetype'] = uniquetype
    emuinfo['numtype'] = numtype
    emuinfo['n'] = fval.shape[0]
    emuinfo['m'] = fval.shape[1]
    # print(np.mean(emuinfo['missingval'],0))
    # print(np.mean(emuinfo['missingval'],1))
    # print(emuinfo['typeval'])
    # print(emuinfo['thetaval'])
    # print(emuinfo['xval'])
    # asdasd

    emulation_setthetacovfunction(emuinfo, emuinfo['corrthetafname'])

    if (emuinfo['thetaval'].shape[0] < (30*emuinfo['thetaval'].shape[1])):
        emuinfo['thetasubset'] = None
    else:
        kskip = (np.floor(emuinfo['thetaval'].shape[0]/(30*emuinfo['thetaval'].shape[1]))).astype(int)
        emuinfo['thetasubset'] = range(0, emuinfo['thetaval'].shape[0], kskip)

    emulation_setxcovfunction(emuinfo, emuinfo['corrxfname'])

    gammasigmasq0 = np.ones(uniquetype.shape[0])
    gammasigmasqLB = np.ones(uniquetype.shape[0])
    gammasigmasqUB = np.ones(uniquetype.shape[0])
    gammamu0 = np.ones(uniquetype.shape[0])
    gammamuLB = np.ones(uniquetype.shape[0])
    gammamuUB = np.ones(uniquetype.shape[0])
    for k in range(0,numtype):
        fconsid = fval[:,np.squeeze(np.where(typeval == uniquetype[k]))]
        meanvalsnow = np.nanmean(fconsid,0)
        # varvalsnow = np.nanvar(fconsid,0)
        gammasigmasq0[k] = 2+np.log(np.nanvar(fconsid))
        gammasigmasqLB[k] = 0 + gammasigmasq0[k]
        gammasigmasqUB[k] = 8 + gammasigmasq0[k]
        gammamu0[k] = np.nanmean(meanvalsnow)
        gammamuLB[k] = np.nanmin(meanvalsnow) - (np.nanmax(meanvalsnow)-np.nanmin(meanvalsnow))
        gammamuUB[k] = np.nanmax(meanvalsnow) + (np.nanmax(meanvalsnow)-np.nanmin(meanvalsnow))
    hypstatparstructure = [emuinfo['gammathetacovhyp0'].shape[0],
                           emuinfo['gammathetacovhyp0'].shape[0]+emuinfo['gammaxcovhyp0'].shape[0],
                           emuinfo['gammathetacovhyp0'].shape[0]+emuinfo['gammaxcovhyp0'].shape[0]+numtype,
                           emuinfo['gammathetacovhyp0'].shape[0]+emuinfo['gammaxcovhyp0'].shape[0]+numtype+numtype]
    emuinfo['gamma0'] = np.concatenate((emuinfo['gammathetacovhyp0'], emuinfo['gammaxcovhyp0'], gammasigmasq0, gammamu0))
    emuinfo['gammaLB'] = np.concatenate((emuinfo['gammathetacovhypLB'], emuinfo['gammaxcovhypLB'], gammasigmasqLB, gammamuLB))
    emuinfo['gammaUB'] = np.concatenate((emuinfo['gammathetacovhypUB'], emuinfo['gammaxcovhypUB'], gammasigmasqUB, gammamuUB))
    emuinfo['hypstatparstructure'] = hypstatparstructure
    emuinfo['gammahat'] = emuinfo['gamma0'][:]
    return emuinfo



def emulation_hypest(emuinfo, modeltype = None):
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
        resid = emuinfo['fval'] - emuinfo['mu']
        mu = emuinfo['mu']
        if emuinfo['ismissingval']:
            fpred = np.ones(emuinfo['fval'].shape)
            donealr = np.zeros(emuinfo['m'])
            for k2 in range(0, emuinfo['m']):
                inds = (np.where(emuinfo['missingval'][:,k2] < 0.5)[0]).astype(int)
                ninds = (np.where(emuinfo['missingval'][:,k2] > 0.5)[0]).astype(int)
                Robs = emuinfo['R'][np.ix_(inds,inds)]
                Rnobsobs = emuinfo['R'][np.ix_(ninds,inds)]
                cholRobs, _ = spla.lapack.dpotrf(Robs,True,True)
                for k in range(k2, emuinfo['m']):
                    if (np.sum(np.abs(emuinfo['missingval'][:,k]-emuinfo['missingval'][:,k2])) < 0.5) and (donealr[k] < 0.5):
                        resvalhalf = spla.solve_triangular(cholRobs, resid[inds,k], lower=True)
                        resvalinv = spla.solve_triangular(cholRobs, resvalhalf, lower=True, trans=True)
                        fpred[ninds,k] = emuinfo['mu'][k] + np.matmul(Rnobsobs,resvalinv)
                        fpred[inds,k] = emuinfo['fval'][inds,k]
                        donealr[k] = 1
            emuinfo['fpred'] = fpred[:]
        else:
            emuinfo['fpred'] = emuinfo['fval'][:]
    #print(np.max(np.abs(emuinfo['fpred'] - emuinfo['fval'])))

    return None

def emulation_myess(x):
    m_chains, n_iters = x.shape
    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))
    post_var = W * (n_iters - 1) / n_iters + B_over_n
    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)
        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0
        t += 1
    return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

def emulation_blockingsubfunc_emg(paramis, inputmis, inputobs, S_inv, R_inv, resid, doDraw = False):
    cholresidvarR = np.linalg.inv(spla.lapack.dpotrf(R_inv[paramis,:][:,paramis],True,True)[0])
    a3 = np.matmul(np.matmul(cholresidvarR,R_inv[paramis,:]), resid)

    a21 = np.matmul(cholresidvarR.T, a3)

    b9 = np.linalg.solve(S_inv[inputmis,:][:,inputmis], S_inv[inputmis,:])
    resid[np.ix_(paramis,inputmis)] -= np.matmul(a21, b9.transpose())

    if doDraw:
        S_inv_22_chol_draw , pr = spla.lapack.dpotrf(np.linalg.inv(0.5*(S_inv[inputmis,:][:,inputmis]+(S_inv[inputmis,:][:,inputmis]).T)),True,True)
        if (pr < 0.5):
                resid[np.ix_(paramis,inputmis)] +=  np.matmul(np.matmul(cholresidvarR,np.random.normal(0,1,(cholresidvarR.shape[1],S_inv_22_chol_draw.shape[1]))),S_inv_22_chol_draw.transpose())
    
    return resid

def emulation_blockingsubfunc(paramis, inputmis, inputobs, Phi, Phi_inv, R_inv, resid, nu, doDraw = False):
    cholresidvarR = np.linalg.inv(spla.lapack.dpotrf(R_inv[paramis,:][:,paramis],True,True)[0])
    a3 = np.matmul(np.matmul(cholresidvarR,R_inv[paramis,:]), resid)

    a21 = np.matmul(cholresidvarR.T, a3)
    Phi_update = Phi - np.matmul(a3.transpose(),a3)
    if (a3.shape[1] > (1.5*(a3.shape[0]-1))):
        a4 = np.matmul(Phi_inv, a3.transpose())
        a8 = np.matmul(a4,np.linalg.solve(np.identity(a4.shape[1]) - np.matmul(a3, a4), a4.transpose()))
        Phi_update_inv = Phi_inv + 0.5*(a8+a8.T)
    else:
        Phi_update_inv = np.linalg.inv(Phi_update)

    b9 = np.linalg.solve(Phi_update_inv[inputmis,:][:,inputmis],Phi_update_inv[inputmis,:])
    resid[np.ix_(paramis,inputmis)] -= np.matmul(a21, b9.transpose())
    if doDraw:
        choltildePhi22now , _ = spla.lapack.dpotrf(Phi_update_inv[inputmis,:][:,inputmis],True,True)
        if (a3.shape[1] < (1.5*(a3.shape[0]-1))):
            Phi_update_11_inv = np.linalg.inv(Phi_update[inputobs,:][:,inputobs])
        else:
            Phi_update_11_inv = Phi_update_inv[inputobs,:][:,inputobs] - np.matmul(Phi_update_inv[inputobs,:][:,inputmis], b9[:,inputobs] )
        Xforsamp = np.matmul(choltildePhi22now, np.random.normal(0,1,(choltildePhi22now.shape[1],nu-paramis.shape[0])))
        Phi_update_22_chol_draw , _ = spla.lapack.dpotrf(np.linalg.inv(np.matmul(Xforsamp, Xforsamp.T)),True,True)

        b5 = np.matmul(np.matmul(a21[:, inputobs], Phi_update_11_inv),a21[:, inputobs].T)
        mat2 = np.matmul(cholresidvarR, spla.lapack.dpotrf(np.identity(a21.shape[0]) + 0.5 * (b5 + b5.T),True,True)[0])
        resid[np.ix_(paramis,inputmis)] +=  np.matmul(np.matmul(mat2,np.random.normal(0,1,(mat2.shape[1],Phi_update_22_chol_draw.shape[1]))),Phi_update_22_chol_draw.transpose())

    a6 = np.matmul(np.matmul(cholresidvarR,R_inv[paramis,:]), resid)
    Phi = Phi_update + np.matmul(a6.transpose(),a6)
    if (a6.shape[1] > (1.5*(a6.shape[0]-1))):
        a10 = np.matmul(Phi_update_inv, a6.transpose())
        a12 = np.matmul(a10,np.linalg.solve(np.identity(a4.shape[1]) + np.matmul(a6, a10), a10.transpose()))
        Phi_inv = Phi_update_inv - 0.5*(a12+a12.T)
    else:
        Phi_inv = np.linalg.inv(Phi)

    return resid, Phi, Phi_inv

def emulation_blocksampling(emuinfo, sampinfo, paramiso, inputmiso, Sinv = None):
    paramis = np.array(np.where(paramiso)[0])
    inputmis = np.array(np.where(inputmiso)[0])
    inputobs = np.array(np.where(inputmiso<0.5)[0])

    if Sinv is None:
        residnew, Phi, Phi_inv = emulation_blockingsubfunc(paramis, inputmis, inputobs, sampinfo['Phi_post'], sampinfo['Phi_post_inv'], emuinfo['R_inv'], sampinfo['fsample'] - emuinfo['mu'], emuinfo['nu_post'], doDraw = True)
        sampinfo['Phi_post'] = Phi
        sampinfo['Phi_post_inv'] = Phi_inv
    else:
        residnew = emulation_blockingsubfunc_emg(paramis, inputmis, inputobs, Sinv, emuinfo['R_inv'], sampinfo['fsample'] - emuinfo['mu'], doDraw = True)
    sampinfo['fsample'] = residnew + emuinfo['mu']

    return sampinfo

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

def emulation_imputeiter(emuinfo, Rs, Cs, tol = 10 ** (-6)):
    tolcutoff = tol*np.mean(np.abs(emuinfo['fpred'][np.where(1-emuinfo['missingval'])]-np.mean(emuinfo['fpred'][np.where(1-emuinfo['missingval'])])))

    for itera in range(0,800):
        fpredbefore = 1*emuinfo['fpred'][np.where(emuinfo['missingval'])]
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
        scalev = np.mean(np.abs(emuinfo['fpred'][np.where(emuinfo['missingval'])]-fpredbefore))
        if (scalev < tolcutoff):
            break
    return emuinfo

def emulation_sampliter(emuinfo, Rs, Cs):
    ndraws = emuinfo['ndrawsamples']
    if ndraws is None:
        ndraws = 200

    if ndraws > 1e5:
        print ('number of samples requested is too large. choose a number under 100000.')
        raise
    else:
        ndrawsactual = np.round(ndraws + 49, -2)
        nsamples =20
        nchains = int(ndrawsactual / nsamples)

    nskip = 1
    fsamp = np.zeros((emuinfo['fpred'].shape[0], emuinfo['fpred'].shape[1], nchains * nsamples))
    pwsamp = np.zeros((emuinfo['fpred'].shape[0], emuinfo['fpred'].shape[1], nchains * nsamples))
    Ssampchol = np.zeros((emuinfo['fpred'].shape[1], emuinfo['fpred'].shape[1], nchains * nsamples))
    for l1 in range(0,nchains):
        if emuinfo['modeltype'] == 'nonparasep':
            sampleinfo = {'Phi_post': 1*emuinfo['Phi_post'],
                          'Phi_post_inv': 1*emuinfo['Phi_post_inv'],
                          'fsample': 1*emuinfo['fpred']
                          }
        else:
            sampleinfo = {'fsample': 1*emuinfo['fpred']
                          }
        for l2 in range(0, nsamples):
            if emuinfo['ismissingval']:
                for itera in range(0,nskip):
                    if emuinfo['modeltype'] == 'parasep':
                        Ssampinv =emuinfo['S_inv']
                        Ssamp = np.linalg.inv(Ssampinv)
                    else:
                        if emuinfo['em_gibbs']:
                            #check if we are still in control
                            normresid = spla.solve_triangular(emuinfo['R_chol'],sampleinfo['fsample'] - emuinfo['fpred']) @ (np.sqrt(emuinfo['nu'] + emuinfo['n']) * np.linalg.cholesky(emuinfo['Phi_post_inv']).T)
                            if np.mean(normresid) > 10:
                                resid = emuinfo['fpred'] - emuinfo['mu']
                            else:
                                resid = sampleinfo['fsample'] - emuinfo['mu']
                            
                            residhalf = spla.solve_triangular(emuinfo['R_chol'], resid, lower=True)
                            
                            Phicholpro = sampleinfo['Phi_post_inv']
                            sampleinfo['Phi_post'] = emuinfo['Phi_prior'] + np.matmul(residhalf.transpose(),residhalf)
                            sampleinfo['Phi_chol_inv'] , _ = spla.lapack.dpotrf(np.linalg.inv(sampleinfo['Phi_post']),True,True)
                            Xforsamp = np.matmul(sampleinfo['Phi_chol_inv'], np.random.normal(0,1,(sampleinfo['Phi_chol_inv'].shape[0],emuinfo['nu_post'])))
                            Ssampinv = np.matmul(Xforsamp,Xforsamp.T)
                        else:
                            resid = sampleinfo['fsample'] - emuinfo['mu']
                            residhalf = spla.solve_triangular(emuinfo['R_chol'], resid, lower=True)
                            Phicholpro = sampleinfo['Phi_post_inv']
                            sampleinfo['Phi_post'] = emuinfo['Phi_prior'] + np.matmul(residhalf.transpose(),residhalf)
                            sampleinfo['Phi_post_inv'] = np.linalg.inv(sampleinfo['Phi_post'])
                    for k in range(0,Rs.shape[0]):
                        if emuinfo['em_gibbs'] or (emuinfo['modeltype'] == 'parasep'):
                            
                            emulation_blocksampling(emuinfo, sampleinfo, Rs[k,:], Cs[k,:], Sinv = Ssampinv)
                        else:
                            try:
                                emulation_blocksampling(emuinfo, sampleinfo, Rs[k,:], Cs[k,:])
                            except:
                                resid = sampleinfo['fsample'] - emuinfo['mu']
                                residhalf = spla.solve_triangular(emuinfo['R_chol'], resid, lower=True)
                                Phicholpro = sampleinfo['Phi_post_inv']
                                sampleinfo['Phi_post'] = emuinfo['Phi_prior'] + np.matmul(residhalf.transpose(),residhalf)
                                sampleinfo['Phi_post_inv'] = np.linalg.inv(sampleinfo['Phi_post'])
                                emulation_blocksampling(emuinfo, sampleinfo, Rs[k,:], Cs[k,:])
                fsamp[:,:,l1*nsamples + l2] = sampleinfo['fsample']
                pwsamp[:,:,l1*nsamples + l2] = spla.solve_triangular(emuinfo['R_chol'], spla.solve_triangular(emuinfo['R_chol'], emuinfo['fpred'] - emuinfo['mu'], lower=True),lower=True, trans=True)

                if emuinfo['em_gibbs'] or (emuinfo['modeltype'] == 'parasep'):
                    Ssampchol[:,:,l1*nsamples + l2], _ = spla.lapack.dpotrf(np.linalg.inv(Ssampinv),True,True)
                else:
                    try:
                        Phi_chol , _ = spla.lapack.dpotrf(sampleinfo['Phi_post_inv'],True,True)
                        Xforsamp = np.matmul(Phi_chol, np.random.normal(0,1,(Phi_chol.shape[0],emuinfo['nu_post'])))
                        Ssampchol[:,:,l1*nsamples + l2] , _ = spla.lapack.dpotrf(np.linalg.inv(np.matmul(Xforsamp,Xforsamp.T)),True,True)
                        Ssampinv =  np.linalg.inv(np.matmul(Xforsamp,Xforsamp.T))
                    except:
                        Ssampchol[:,:,l1*nsamples + l2] = sampleinfo['Phi_post'] / sampleinfo['nu_post']

            else:
                fsamp[:,:,l1*nsamples + l2] = emuinfo['fpred']
                pwsamp[:,:,l1*nsamples + l2] = spla.solve_triangular(emuinfo['R_chol'], spla.solve_triangular(emuinfo['R_chol'], emuinfo['fpred'] - emuinfo['mu'], lower=True),lower=True, trans=True)
                Xforsamp = np.linalg.solve(emuinfo['Phi_post_chol'].T, np.random.normal(0,1,(emuinfo['Phi_post_chol'].shape[0],emuinfo['nu_post'])))
                Ssampchol[:,:,l1*nsamples + l2] = np.linalg.inv(spla.lapack.dpotrf(np.matmul(Xforsamp,Xforsamp.T),True,True)[0]).T

    emuinfo['pwsample'] = pwsamp
    emuinfo['fsample'] = fsamp
    emuinfo['Ssamplechol'] = Ssampchol
    return emuinfo

def emulation_matrixblockerfunction(emuinfo):
    missingmat = emuinfo['missingval']
    Rs = np.zeros((np.int(np.sum(missingmat)),missingmat.shape[0]))
    Cs = np.zeros((np.int(np.sum(missingmat)),missingmat.shape[1]))
    #breakintoblocks
    ivaltr = np.where(np.sum(missingmat,1))[0]
    jvaltr = np.where(np.sum(missingmat,0))[0]
    missingmat = missingmat[np.ix_(ivaltr,jvaltr)]
    k = 0
    while(np.sum(missingmat) > 0.5):
        ival = np.where(np.sum(missingmat,1))[0]
        jval = np.where(np.sum(missingmat,0))[0]
        ivaltr = ivaltr[ival]
        jvaltr = jvaltr[jval]
        missingmat = missingmat[np.ix_(ival,jval)]
        n = missingmat.shape[0]
        if emuinfo['blocking'] == 'individual':
            istar = 0
            jstar = np.where(missingmat[0,:])[0][0]
            Rs[k,ivaltr[istar]] = 1
            Cs[k,jvaltr[jstar]] = 1
            missingmat[istar,jstar] = 0
            if n == 1:
                k = k + 1
                break

        if emuinfo['blocking'] == 'row':
            if n == 1:
                Rs[k,ivaltr] = 1
                Cs[k,jvaltr] = 1
                k = k + 1
                break
            blockrow = np.zeros(missingmat.shape[0])
            blockrow[0] = 1
            blockcol = missingmat[0,:]
            Rs[k,ivaltr[np.where(blockrow)[0]]] = 1
            Cs[k,jvaltr[np.where(blockcol)[0]]] = 1
            missingmat[np.ix_(np.where(blockrow)[0],np.where(blockcol)[0])] = 0
        if emuinfo['blocking'] == 'column':
            if n == 1:
                Rs[k,ivaltr] = 1
                Cs[k,jvaltr] = 1
                k = k + 1
                break
            blockcol = np.zeros(missingmat.shape[1])
            blockcol[0] = 1
            blockrow = missingmat[:,0]
            Rs[k,ivaltr[np.where(blockrow)[0]]] = 1
            Cs[k,jvaltr[np.where(blockcol)[0]]] = 1
            missingmat[np.ix_(np.where(blockrow)[0],np.where(blockcol)[0])] = 0
        else:
            if n == 1:
                Rs[k,ivaltr] = 1
                Cs[k,jvaltr] = 1
                k = k + 1
                break
            blockrow = np.zeros(missingmat.shape[0])
            blockrow[0] = 1
            blockcol = missingmat[0,:]

            numrow = np.sum(blockrow)
            numcol = np.sum(blockcol)

            newsize = np.zeros(n)
            for j in range(0,n):
                newsize = 0*newsize
                for i in range(0,n):
                    if(blockrow[i] == 0):
                        numcolpot = np.sum(blockcol*missingmat[i,:])
                        newsize[i] =  numcolpot*(numrow+1)
                    else:
                        newsize[i] = 0
                if((numrow*numcol+0.5) < np.max(newsize)):
                    istar = np.argmax(newsize)
                    blockrow[istar] = 1
                    blockcol = blockcol*missingmat[istar,:]
                    numrow = numrow+1
                    numcol = np.sum(blockcol)
                else:
                    break
            Rs[k,ivaltr[np.where(blockrow)[0]]] = 1
            Cs[k,jvaltr[np.where(blockcol)[0]]] = 1
            missingmat[np.ix_(np.where(blockrow)[0],np.where(blockcol)[0])] = 0
        k =k + 1
    Rs = Rs[:k,]
    Cs = Cs[:k,]
    return Rs, Cs