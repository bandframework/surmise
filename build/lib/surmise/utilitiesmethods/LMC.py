import numpy as np
import scipy.optimize as spo

'''
Metropolis-adjusted Langevin algorithm or Langevin Monte Carlo (LMC)
'''


def sampler(logpostfunc, options):
    '''

    Parameters
    ----------
    logpostfunc : function
        A function call describing the log of the posterior distribution.
    options : dict
        a dictionary contains the output of the sampler.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        Matrix of sampled parameter values

    '''

    if 'theta0' in options.keys():
        theta0 = options['theta0']
    else:
        raise ValueError('Unknown theta0')

    # Initialize
    if 'numsamp' in options.keys():
        numsamp = options['numsamp']
    else:
        numsamp = 2000

    # Minimum effective sample size (ESS) desired in the returned samples
    tarESS = np.max((150, 10 * theta0.shape[1]))

    # Test
    testout = logpostfunc(theta0[0:2, :])
    if type(testout) is tuple:
        if len(testout) != 2:
            raise ValueError('log density does not return 1 or 2 elements')
        if testout[1].shape[1] is not theta0.shape[1]:
            raise ValueError('derivative appears to be the wrong shape')

        logpostf = logpostfunc

        def logpostf_grad(theta):
            return logpostfunc(theta)[1]

        try:
            testout = logpostfunc(theta0[10, :], return_grad=False)
            if type(testout) is tuple:
                raise ValueError('Cannot stop returning a grad')

            def logpostf_nograd(theta):
                return logpostfunc(theta, return_grad=False)

        except Exception:
            def logpostf_nograd(theta):
                return logpostfunc(theta)[0]
    else:
        logpostf_grad = None
        logpostf = logpostfunc
        logpostf_nograd = logpostfunc

    if logpostf_grad is None:
        rho = 2 / theta0.shape[1] ** (1/2)
        taracc = 0.25
    else:
        rho = 2 / theta0.shape[1] ** (1/6)
        taracc = 0.60

    keepgoing = True
    theta0 = np.unique(theta0, axis=0)
    iteratttempt = 0
    while keepgoing:
        logpost = logpostf_nograd(theta0)/4
        mlogpost = np.max(logpost)
        logpost -= (mlogpost + np.log(np.sum(np.exp(logpost - mlogpost))))
        post = np.exp(logpost)
        post = post/np.sum(post)
        thetaposs = theta0[np.random.choice(range(0, theta0.shape[0]),
                                            size=1000,
                                            p=post.reshape((theta0.shape[0],
                                                            ))), :]

        if np.any(np.std(thetaposs, 0) < 10 ** (-8) * np.min(np.std(theta0,
                                                                    0))):
            thetastar = theta0[np.argmax(logpost), :]
            theta0 = thetastar + (theta0 - thetastar) / 2
            iteratttempt += 1
        else:
            theta0 = thetaposs
            keepgoing = False
        if iteratttempt > 10:
            keepgoing = False

    thetaop = theta0[:10, :]
    thetac = np.mean(theta0, 0)
    thetas = np.maximum(np.std(theta0, 0), 10 ** (-4) * np.std(theta0))

    def neglogpostf_nograd(thetap):
        theta = thetac + thetas * thetap
        return -logpostf_nograd(theta.reshape((1, len(theta))))[0]

    if logpostf_grad is not None:
        def neglogpostf_grad(thetap):
            theta = thetac + thetas * thetap
            return -thetas * logpostf_grad(theta.reshape((1, len(theta))))

    boundL = np.maximum(-10*np.ones(theta0.shape[1]),
                        np.min((theta0 - thetac)/thetas, 0))
    boundU = np.minimum(10*np.ones(theta0.shape[1]),
                        np.max((theta0 - thetac)/thetas, 0))
    bounds = spo.Bounds(boundL, boundU)

    keeptryingwithgrad = True
    failureswithgrad = 0

    for k in range(0, thetaop.shape[0]):
        theta0 = (thetaop[k, :] - thetac) / thetas
        if logpostf_grad is None:
            opval = spo.minimize(neglogpostf_nograd,
                                 theta0,
                                 method='L-BFGS-B',
                                 bounds=bounds,
                                 options={'maxiter': 4, 'maxfun': 100})
            thetaop[k, :] = thetac + thetas * opval.x

        else:
            if keeptryingwithgrad:
                opval = spo.minimize(neglogpostf_nograd,
                                     theta0,
                                     method='L-BFGS-B',
                                     jac=neglogpostf_grad,
                                     bounds=bounds,
                                     options={'maxiter': 15, 'maxfun': 100})
                thetaop[k, :] = thetac + thetas * opval.x

            if not keeptryingwithgrad or not opval.success:
                if keeptryingwithgrad:
                    failureswithgrad += 1
                    alpha = failureswithgrad + 0.25
                    beta = (k - failureswithgrad + 1)
                    stdtest = np.sqrt(alpha * beta / ((alpha + beta + 1) *
                                                      ((alpha + beta)**2)))
                    meantest = alpha/(alpha + beta)
                    if meantest - 3*stdtest > 0.25:
                        keeptryingwithgrad = False

                opval = spo.minimize(neglogpostf_nograd,
                                     theta0,
                                     method='L-BFGS-B',
                                     bounds=bounds,
                                     options={'maxiter': 4, 'maxfun': 100})
                thetaop[k, :] = thetac + thetas * opval.x

    for k in range(0, thetaop.shape[0]):
        LB, UB = test1dboundarys(thetaop[k, :], logpostf_nograd, thetas)
        thetas = np.maximum(thetas, 0.5 * (UB - LB))
        theta0 = np.vstack((theta0, LB))
        theta0 = np.vstack((theta0, UB))

    thetasave = theta0
    Lsave = logpostf_nograd(thetasave)

    tau = -1
    rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
    numchain = 50
    maxiters = 10
    numsamppc = 10
    for iters in range(0, maxiters):
        Lsave = np.squeeze(np.reshape(Lsave, (-1, 1)))
        mLsave = np.max(Lsave)
        Lsave -= mLsave + np.log(np.sum(np.exp(Lsave - mLsave)))
        post = np.exp(Lsave/4)
        post = post/np.sum(post)
        startingv = np.random.choice(np.arange(0, Lsave.shape[0]),
                                     size=Lsave.shape[0], p=post)
        thetasave = thetasave[startingv, :]
        covmat0 = np.cov(thetasave.T)

        if covmat0.ndim > 1:
            covmat0 += (10 ** (-4)) * np.diag(np.diag(covmat0) + thetas)
            Wc, Vc = np.linalg.eigh(covmat0)
            hc = (Vc @ np.diag(np.sqrt(Wc)) @ Vc.T)
        else:
            hc = np.sqrt(covmat0 + thetas)

        thetac = thetasave[np.random.choice(range(0, thetasave.shape[0]),
                                            size=numchain), :]

        if logpostf_grad is not None:
            fval, dfval = logpostf(thetac)
        else:
            fval = logpostf_nograd(thetac)

        thetasave = np.zeros((numchain, numsamppc, thetac.shape[1]))
        Lsave = np.zeros((numchain, numsamppc))
        numtimes = 0

        for k in range(0, numsamppc):
            rvalo = np.random.normal(0, 1, thetac.shape)
            rval = np.sqrt(2) * rho * (rvalo @ hc)

            if rval.ndim != thetac.ndim:
                rval = np.reshape(rval, (thetac.shape))
            thetap = thetac + rval

            if logpostf_grad is not None:
                diffval = rho ** 2 * (dfval @ covmat0)
                thetap += diffval
                fvalp, dfvalp = logpostf(thetap)
                term1 = rvalo / np.sqrt(2)
                term2 = (dfval + dfvalp) @ hc * rho / 2
                qadj = -(2 * np.sum(term1 * term2, 1) + np.sum(term2**2, 1))
            else:
                fvalp = logpostf_nograd(thetap)
                qadj = np.zeros(fvalp.shape)

            swaprnd = np.log(np.random.uniform(size=fval.shape[0]))
            whereswap = np.where(swaprnd < (fvalp - fval + qadj))[0]

            if whereswap.shape[0] > 0:
                numtimes = numtimes + (whereswap.shape[0]/numchain)
                thetac[whereswap, :] = 1*thetap[whereswap, :]
                fval[whereswap] = 1*fvalp[whereswap]

                if logpostf_grad is not None:
                    dfval[whereswap, :] = 1*dfvalp[whereswap, :]

            # Robbins-Monroe updates
            if iters < 1.5:
                tau = tau + 1/np.sqrt(1 + 100/numchain * k) * \
                    ((whereswap.shape[0]/numchain) - taracc)
                rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
            thetasave[:, k, :] = thetac
            Lsave[:, k] = fval.reshape((len(fval), ))

        mut = np.mean(np.mean(thetasave, 1), 0)
        B = np.zeros(mut.shape)
        autocorr = np.zeros(mut.shape)
        W = np.zeros(mut.shape)
        for i in range(0, numchain):
            muv = np.mean(thetasave[i, :, :], 0)
            autocorr += 1/numchain * \
                np.mean((thetasave[i, 0:(numsamppc - 1), :] - muv.T) *
                        (thetasave[i, 1:, :] - muv.T), 0)
            W += 1/numchain * \
                np.mean((thetasave[i, 0:(numsamppc-1), :] - muv.T)**2, 0)
            B += numsamppc/(numchain - 1) * ((muv - mut)**2)
        varplus = W + 1/numsamppc * B

        if np.any(varplus < 10**(-10)):
            raise ValueError('Sampler failed to move at all.')
        else:
            rhohat = (1 - (W - autocorr)/varplus)

        ESS = 1 + numchain*numsamppc*(1 - np.abs(rhohat))
        thetasave = np.reshape(thetasave, (-1, thetac.shape[1]))
        accr = numtimes/numsamppc

        if iters > 1.5 and accr > 0.1 and accr < 0.9 and\
                (np.mean(ESS) > tarESS):
            break
        elif accr < taracc*4/5 or accr > taracc*5/4:
            tau = tau + 1/(1 + iters) * (accr - taracc)
            rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
        if accr < taracc*1.5 and accr > taracc*0.6:
            trm = np.min((1.5*tarESS/np.mean(ESS), 4))
            numsamppc = np.ceil(numsamppc*trm).astype('int')

    theta = thetasave[np.random.choice(range(0, thetasave.shape[0]),
                                       size=numsamp), :]
    sampler_info = {'theta': theta}
    return sampler_info


def test1dboundarys(theta0, logpostfunchere, thetas):
    L0 = logpostfunchere(theta0.reshape((1, len(theta0))))
    thetaminsave = np.zeros(theta0.shape)
    thetamaxsave = np.zeros(theta0.shape)
    epsnorm = 1
    for k in range(0, theta0.shape[0]):
        notfarenough = 0
        farenough = 0
        eps = epsnorm * thetas[k]
        keepgoing = True
        while keepgoing:
            thetaadj = 1 * theta0
            thetaadj[k] += eps
            L1 = logpostfunchere(thetaadj.reshape((1, len(thetaadj))))
            if (L0-L1) < 3:
                eps = eps*2
                notfarenough += 1
            else:
                eps = eps/2
                farenough += 1
                thetamaxsave[k] = 1 * thetaadj[k]
            if notfarenough > 1.5 and farenough > 1.5:
                keepgoing = False
                epsnorm = eps/thetas[k]
        notfarenough = 0
        farenough = 0
        keepgoing = True
        while keepgoing:
            thetaadj = 1 * theta0
            thetaadj[k] -= eps
            L1 = logpostfunchere(thetaadj.reshape((1, len(thetaadj))))
            if (L0-L1) < 3:
                eps = eps*2
                notfarenough += 1
            else:
                eps = eps/2
                farenough += 1
                thetaminsave[k] = 1 * thetaadj[k]
            if notfarenough > 1.5 and farenough > 1.5:
                keepgoing = False
                epsnorm = eps/thetas[k]
    return thetaminsave, thetamaxsave
