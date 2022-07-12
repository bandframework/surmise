
import numpy as np
import scipy.linalg as spla
import scipy.optimize as spo



def emulation_likderivativetester(fname, dfname, emuinfo, thetasubset = None):
    gammav = (emuinfo['gammaUB'] + emuinfo['gammaLB']) / 2
    gammav = (emuinfo['gamma0'])
    L0 = fname(gammav, emuinfo)
    dL0 = dfname(gammav, emuinfo)

    for k in range(0,emuinfo['gamma0'].shape[0]):
        gammap = 1*gammav[:]
        gammap[k] += 10 ** (-6)
        L1 = fname(gammap, emuinfo)
        print((L1-L0) * (10 ** 6))
        print(dL0[k])

    return None

def emulation_lik_indp(gammav, emuinfo, fval = None):
    if emuinfo['thetasubset'] is None:
        R = emulation_getR(emuinfo, gammav, False)
    else:
        R = emulation_getR(emuinfo, gammav, False, thetasubset = emuinfo['thetasubset'])
    (cholR, pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        print('here ind')
        return float("inf")

    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        if emuinfo['thetasubset'] is None:
            resid = emuinfo['fval'] - mu
            missval = emuinfo['missingval']
        else:
            resid = emuinfo['fval'][emuinfo['thetasubset'], :] - mu
            missval = emuinfo['missingval'][emuinfo['thetasubset'],:]
    else:
        if emuinfo['thetasubset'] is None:
            resid = fval - mu
            missval = emuinfo['missingval']
        else:
            resid = fval[emuinfo['thetasubset'], :] - mu
            missval = emuinfo['missingval'][emuinfo['thetasubset'],:]


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
            if (np.sum(np.abs(missval[:,k]-missval[:,k2])) < 0.5) and (donealr[k] < 0.5):
                resvalhalf = spla.solve_triangular(cholRobs, resid[inds,k], lower=True)
                logs2sum += (nhere/2+emuinfo['nu']) * np.log(1/2 * np.sum(resvalhalf * resvalhalf) + emuinfo['nu']*sigma2[k]) - emuinfo['nu']*np.log(emuinfo['nu']*sigma2[k])
                logdetRsum += logdetval
                donealr[k] = 1
    gammanorm = (gammav-emuinfo['gamma0'])/(emuinfo['gammaUB']-emuinfo['gammaLB'])
    loglik = logs2sum + 0.5 * logdetRsum + 8*np.sum(gammanorm ** 2)
    return loglik

def emulation_dlik_indp(gammav, emuinfo, fval = None):
    if emuinfo['thetasubset'] is None:
        R, dR = emulation_getR(emuinfo, gammav, True)
    else:
        R, dR = emulation_getR(emuinfo, gammav, True, thetasubset = emuinfo['thetasubset'])
    (cholR, pd) = spla.lapack.dpotrf(R, True, True)
    if pd > 0.5:
        print('here ind')
        return float("inf")

    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        if emuinfo['thetasubset'] is None:
            resid = emuinfo['fval'] - mu
            missval = emuinfo['missingval']
        else:
            resid = emuinfo['fval'][emuinfo['thetasubset'], :] - mu
            missval = emuinfo['missingval'][emuinfo['thetasubset'],:]
    else:
        if emuinfo['thetasubset'] is None:
            resid = fval - mu
            missval = emuinfo['missingval']
        else:
            resid = fval[emuinfo['thetasubset'], :] - mu
            missval = emuinfo['missingval'][emuinfo['thetasubset'],:]


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
            if (np.sum(np.abs(missval[:,k]-missval[:,k2])) < 0.5) and (donealr[k] < 0.5):
                dlogdetsum = dlogdetsum + addterm
                resvalhalf = spla.solve_triangular(cholRobs, resid[inds,k], lower=True)
                resvalinv = spla.solve_triangular(cholRobs, resvalhalf, lower=True, trans=True)
                s2calc = (1/2 * np.sum(resvalhalf * resvalhalf) + emuinfo['nu'] * sigma2[k])
                for l in range(0,emuinfo['hypstatparstructure'][0]):
                    dnum = - resvalinv.T @ dR[inds,:,l][:,inds] @ resvalinv
                    dlogs2sum[l] =   dlogs2sum[l] +  (1/2* (nhere/2+emuinfo['nu'] ) * dnum) / s2calc
                for l in range(0,emuinfo['numtype']):
                     if (emuinfo['typeval'][k] == emuinfo['uniquetype'][l]):
                         dlogs2sum[emuinfo['hypstatparstructure'][1] + l] += emuinfo['nu'] * (((nhere/2+emuinfo['nu'] ) * sigma2[k]) / s2calc-1)
                         dnum = - 2 * np.sum(resvalinv)
                         dlogs2sum[emuinfo['hypstatparstructure'][2] + l] += (1/2* (nhere/2+emuinfo['nu'] ) * dnum) / s2calc
                donealr[k] = 1
    dloglik = dlogs2sum + 0.5 * dlogdetsum + 16 * (gammav-emuinfo['gamma0'])/((emuinfo['gammaUB']-emuinfo['gammaLB']) ** 2)
    return dloglik

#
# def emulation_lik_nonparasep(gammav, emuinfo, fval = None):
#     if emuinfo['thetasubset'] is None:
#         R = emulation_getR(emuinfo, gammav, False)
#     else:
#         R = emulation_getR(emuinfo, gammav, False, thetasubset = emuinfo['thetasubset'])
#
#
#     mu = emulation_getmu(emuinfo, gammav, False)
#     sigma2 = emulation_getsigma2(emuinfo, gammav, False)
#     if fval is None:
#         if emuinfo['thetasubset'] is None:
#             resid = emuinfo['fpred'] - mu
#         else:
#             resid = emuinfo['fpred'][emuinfo['thetasubset'], :] - mu
#     else:
#         if emuinfo['thetasubset'] is None:
#             resid = fval - mu
#         else:
#             resid = fval[emuinfo['thetasubset'], :] - mu
#
#     (cholR ,pd) = spla.lapack.dpotrf(R, True, True)
#     if pd > 0.5:
#         return float("inf")
#
#     nhere = resid.shape[0]
#     logdetR = 2*np.sum(np.log(np.diag(cholR)))
#     residhalf = spla.solve_triangular(cholR, resid, lower=True)
#     Sigmapart1 = emulation_getS(emuinfo, gammav, withdir = False)
#     Sigma = emuinfo['nu'] * (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
#     (cholSigma ,pd) = spla.lapack.dpotrf(Sigma, True, True)
#     if pd > 0.5:
#         return float("inf")
#     logdetSigma = 2*np.sum(np.log(np.diag(cholSigma)))
#     Sigmapost = residhalf.T @ residhalf + Sigma
#     (cholSigmapost ,pd) = spla.lapack.dpotrf(Sigmapost, True, True)
#     if pd < 0.5:
#         logdetSigmapost = 2*np.sum(np.log(np.diag(cholSigmapost)))
#         gammanorm = (gammav-emuinfo['gamma0'])/(emuinfo['gammaUB']-emuinfo['gammaLB'])
#         loglik = (nhere+emuinfo['nu'])*logdetSigmapost - emuinfo['nu']*logdetSigma + emuinfo['m']*logdetR + 8*np.sum(gammanorm ** 2)
#         return loglik
#     else:
#         return float("inf")
#
# def emulation_dlik_nonparasep(gammav, emuinfo, fval = None):
#     if emuinfo['thetasubset'] is None:
#         R, dR = emulation_getR(emuinfo, gammav, True)
#     else:
#         R, dR = emulation_getR(emuinfo, gammav, True, thetasubset = emuinfo['thetasubset'])
#
#     mu = emulation_getmu(emuinfo, gammav, False)
#     sigma2 = emulation_getsigma2(emuinfo, gammav, False)
#     if fval is None:
#         if emuinfo['thetasubset'] is None:
#             resid = emuinfo['fpred'] - mu
#         else:
#             resid = emuinfo['fpred'][emuinfo['thetasubset'], :] - mu
#     else:
#         if emuinfo['thetasubset'] is None:
#             resid = fval - mu
#         else:
#             resid = fval[emuinfo['thetasubset'], :] - mu
#
#     (cholR ,pd) = spla.lapack.dpotrf(R, True, True)
#     if pd > 0.5:
#         return float("inf")
#
#     nhere = resid.shape[0]
#     invR = spla.solve_triangular(cholR,spla.solve_triangular(cholR,np.diag(np.ones(resid.shape[0])), lower=True), lower=True, trans=True)
#     Sigmapart1, dSigmapart1 = emulation_getS(emuinfo, gammav, withdir = True)
#     Sigma = emuinfo['nu'] * (np.diag(np.sqrt(sigma2)) @ Sigmapart1 @ np.diag(np.sqrt(sigma2)))
#     residhalf = spla.solve_triangular(cholR, resid, lower=True)
#     residinv = spla.solve_triangular(cholR, residhalf, lower=True, trans=True)
#     (cholSigma ,pd) = spla.lapack.dpotrf(Sigma, True, True)
#     if pd > 0.5:
#         return float("inf")
#     invSigma = spla.solve_triangular(cholSigma,spla.solve_triangular(cholSigma,np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)
#     Sigmapost = residhalf.T @ residhalf + Sigma
#     (cholSigmapost ,pd) = spla.lapack.dpotrf(Sigmapost, True, True)
#     if pd < 0.5:
#         invSigmapost = spla.solve_triangular(cholSigmapost,spla.solve_triangular(cholSigmapost,np.diag(np.ones(emuinfo['m'])), lower=True), lower=True, trans=True)
#         dlogdetR = np.zeros(gammav.shape)
#         dlogdetSigma = np.zeros(gammav.shape)
#         dlogdetSigmapost = np.zeros(gammav.shape)
#         A5 = residinv @ invSigmapost @ residinv.T
#         A12 = np.diag(np.sqrt(sigma2)) @ invSigmapost @ np.diag(np.sqrt(sigma2))
#         A11 = np.diag(np.sqrt(sigma2)) @ invSigma @ np.diag(np.sqrt(sigma2))
#         for k in range(0,emuinfo['hypstatparstructure'][0]):
#             dlogdetR[k] = np.sum(invR * np.squeeze(dR[:,:,k]))
#             dlogdetSigmapost[k] = -np.sum(A5 * np.squeeze(dR[:,:,k]))
#         for k in range(emuinfo['hypstatparstructure'][0],emuinfo['hypstatparstructure'][1]):
#             A10 = np.squeeze(dSigmapart1[:,:, k- emuinfo['hypstatparstructure'][0]])
#             dlogdetSigmapost[k] = emuinfo['nu'] *np.sum(A12 * A10)
#             dlogdetSigma[k] = emuinfo['nu'] *np.sum(A11 * A10)
#         for k in range(emuinfo['hypstatparstructure'][1],emuinfo['hypstatparstructure'][2]):
#             typevalnow = k-emuinfo['hypstatparstructure'][1]
#             Dhere = np.zeros(emuinfo['m'])
#             Dhere[emuinfo['typeval'] == emuinfo['uniquetype'][typevalnow]] = 1
#             A3 = emuinfo['nu']/2*(np.diag((Dhere/np.sqrt(sigma2))) @ Sigmapart1 @ np.diag((np.sqrt(sigma2))) + np.diag((np.sqrt(sigma2))) @ Sigmapart1 @ np.diag((Dhere/np.sqrt(sigma2))))
#             dlogdetSigmapost[k] = np.sum(invSigmapost*A3) * np.exp(gammav[emuinfo['hypstatparstructure'][1]+typevalnow])
#             dlogdetSigma[k] = np.sum(invSigma*A3) * np.exp(gammav[emuinfo['hypstatparstructure'][1]+typevalnow])
#             dresidval = spla.solve_triangular(cholR, np.squeeze(0*resid + Dhere), lower=True)
#             A4 = dresidval.T @ residhalf + residhalf.T @ dresidval
#             dlogdetSigmapost[k +emuinfo['hypstatparstructure'][2] - emuinfo['hypstatparstructure'][1]] = -np.sum(invSigmapost * A4)
#         dloglik = (nhere+emuinfo['nu'])*dlogdetSigmapost - emuinfo['nu']*dlogdetSigma + emuinfo['m']*dlogdetR +  16 * (gammav-emuinfo['gamma0'])/((emuinfo['gammaUB']-emuinfo['gammaLB']) ** 2)
#         return dloglik
#     else:
#         return float("inf")
#

def emulation_lik_parasep(gammav, emuinfo, fval = None):
    if emuinfo['thetasubset'] is None:
        R = emulation_getR(emuinfo, gammav, False)
    else:
        R = emulation_getR(emuinfo, gammav, False, thetasubset = emuinfo['thetasubset'])
    
    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        if emuinfo['thetasubset'] is None:
            resid = emuinfo['fpred'] - mu
        else:
            resid = emuinfo['fpred'][emuinfo['thetasubset'], :] - mu
    else:
        if emuinfo['thetasubset'] is None:
            resid = fval - mu
        else:
            resid = fval[emuinfo['thetasubset'], :] - mu
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
    if emuinfo['thetasubset'] is None:
        R, dR = emulation_getR(emuinfo, gammav, True)
    else:
        R, dR = emulation_getR(emuinfo, gammav, True, thetasubset = emuinfo['thetasubset'])
    
    mu = emulation_getmu(emuinfo, gammav, False)
    sigma2 = emulation_getsigma2(emuinfo, gammav, False)
    if fval is None:
        if emuinfo['thetasubset'] is None:
            resid = emuinfo['fpred'] - mu
        else:
            resid = emuinfo['fpred'][emuinfo['thetasubset'], :] - mu
    else:
        if emuinfo['thetasubset'] is None:
            resid = fval - mu
        else:
            resid = fval[emuinfo['thetasubset'], :] - mu
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
            Dhere[emuinfo['typeval'] == emuinfo['uniquetype'][typevalnow]] = 1
            ddosomestuff[k+emuinfo['numtype']] = - 2 * np.sum((resid.T @ invR @ (0*resid + Dhere)) * invSigma)
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
        return emuinfo['covthetaf'](emuinfo['thetaval'], emuinfo['thetaval'], gammathetacovhyp, returndir = withdir, diffX = diffTheta, sameX = sameTheta)
    else:
        return emuinfo['covthetaf'](emuinfo['thetaval'][thetasubset,:], emuinfo['thetaval'][thetasubset,:], gammathetacovhyp, returndir = withdir, diffX = diffTheta, sameX = sameTheta)

def emulation_getmu(emuinfo, gammav = None, withdir = False):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammamuhyp = gammav[(emuinfo['hypstatparstructure'][2]):]
    mu = np.ones(emuinfo['m'])
    for k in range(0,gammamuhyp.shape[0]):
        mu[emuinfo['typeval'] == emuinfo['uniquetype'][k]] = gammamuhyp[k]

    return mu

def emulation_getsigma2(emuinfo, gammav = None, withdir = False):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammasigmasqhyp = gammav[(emuinfo['hypstatparstructure'][1]):(emuinfo['hypstatparstructure'][2])]
    sigma2 = np.ones(emuinfo['m'])
    for k in range(0,gammasigmasqhyp.shape[0]):
        sigma2[emuinfo['typeval'] == emuinfo['uniquetype'][k]] = np.exp(gammasigmasqhyp[k])

    return sigma2

def emulation_getS(emuinfo, gammav = None, withdir = False):
    if gammav is None:
        gammav = emuinfo['gammahat']

    gammaxcovhyp = gammav[(emuinfo['hypstatparstructure'][0]):(emuinfo['hypstatparstructure'][1])]
    if withdir:
        return emuinfo['covxf'](emuinfo['xval'], emuinfo['xval'], gammaxcovhyp ,type1 = emuinfo['typeval'], type2 = emuinfo['typeval'], returndir = True)
    else:
        return emuinfo['covxf'](emuinfo['xval'], emuinfo['xval'], gammaxcovhyp ,type1 = emuinfo['typeval'], type2 = emuinfo['typeval'], returndir = False)