import numpy as np
import scipy.optimize as spo

'''
Parallel-Tempering Ensemble MCMC
'''


def sampler(logpostfunc,
            draw_func,
            numsamp=2000,
            theta0=None,
            maxtemp=30,
            numtemps=32,
            numchain=16,
            fractunning=0.5,
            sampperchain=400,
            **ptlmc_options):
    '''

    Parameters
    ----------
    logpost_func : function
        A function call describing the log of the posterior distribution.
            If no gradient, logpostfunc should take a value of an m by p numpy
            array of parameters and theta and return
            a length m numpy array of log posterior evaluations.
            If gradient, logpostfunc should return a tuple.  The first element
            in the tuple should be as listed above.
            The second element in the tuple should be an m by p matrix of
            gradients of the log posterior.
    draw_func : TYPE
        DESCRIPTION.
    numsamp : TYPE, optional
        number of samples you want from the posterior. The default is 2000.
    theta0 : TYPE, optional
        DESCRIPTION. The default is None.
    maxtemp : TYPE, optional
        DESCRIPTION. The default is 30.
    numtemps : TYPE, optional
        DESCRIPTION. The default is 32.
    numchain : TYPE, optional
        DESCRIPTION. The default is 16.
    fractunning : TYPE, optional
        DESCRIPTION. The default is 0.5.
    sampperchain : TYPE, optional
        DESCRIPTION. The default is 400.
    numopt : TYPE, optional
        DESCRIPTION. The default is 10.
    **ptlmc_options : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    if theta0 is None:
        theta0 = draw_func(1000)

    if theta0.shape[0] < 10*theta0.shape[1]:
        raise ValueError('Supply more initial thetas!')

    samptunning = np.ceil(sampperchain*fractunning).astype('int')
    totnumchain = numtemps+numchain
    temps = np.concatenate((np.exp(np.linspace(np.log(maxtemp),
                                               np.log(maxtemp)/(numtemps+1),
                                               numtemps)),
                            np.ones(numchain)))  # ratio idea tend from emcee
    temps = np.array(temps, ndmin=2).T
    numopt = temps.shape[0]
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

    ord1 = np.argsort(-np.squeeze(logpostf_nograd(theta0)) +
                      (theta0.shape[1] *
                       np.random.standard_normal(size=theta0.shape[0])**2))
    theta0 = theta0[ord1[0:totnumchain], :]

    # begin preoptimizer
    thetacen = np.mean(theta0, 0)
    thetas = np.maximum(np.std(theta0, 0), 10 ** (-8) * np.std(theta0))

    def neglogpostf_nograd(thetap):
        theta = thetacen + thetas * thetap
        return -logpostf_nograd(theta.reshape((1, len(theta))))[0]
    if logpostf_grad is not None:
        def neglogpostf_grad(thetap):
            theta = thetacen + thetas * thetap
            return -thetas * logpostf_grad(theta.reshape((1, len(theta))))
    boundL = np.maximum(-10*np.ones(theta0.shape[1]),
                        np.min((theta0 - thetacen)/thetas, 0))
    boundU = np.minimum(10*np.ones(theta0.shape[1]),
                        np.max((theta0 - thetacen)/thetas, 0))
    bounds = spo.Bounds(boundL, boundU)
    thetaop = theta0
    for k in range(0, numopt):
        if logpostf_grad is None:
            opval = spo.minimize(neglogpostf_nograd,
                                 (thetaop[k, :] - thetacen) / thetas,
                                 method='L-BFGS-B',
                                 bounds=bounds)
            thetaop[k, :] = thetacen + thetas * opval.x
        else:
            opval = spo.minimize(neglogpostf_nograd,
                                 (thetaop[k, :] - thetacen) / thetas,
                                 method='L-BFGS-B',
                                 jac=neglogpostf_grad,
                                 bounds=bounds)
            thetaop[k, :] = thetacen + thetas * opval.x
        W, V = np.linalg.eigh(opval.hess_inv @ np.eye(thetacen.shape[0]))
        notmoved = True
        if k == 0:
            notmoved = False
        stepadj = 4
        l0 = neglogpostf_nograd(opval.x)
        while notmoved:
            r = (V.T*np.sqrt(W)) @ (V @
                 np.random.standard_normal(size=thetacen.shape[0]))
            if (neglogpostf_nograd((stepadj * r + opval.x)) -
                    l0) < 3*thetacen.shape[0]:
                thetaop[k, :] = thetacen + thetas * (stepadj * r + opval.x)
                notmoved = False
            else:
                stepadj /= 2
            if stepadj < 1/16:
                thetaop[k, :] = thetacen + thetas * (opval.x)
                notmoved = False
    # end Preoptimizer
    thetac = thetaop
    # done shrink
    if logpostf_grad is not None:
        fval, dfval = logpostf(thetac)
        fval = fval/temps
        dfval = dfval/temps
    else:
        fval = logpostf_nograd(thetac)
        fval = fval/temps

    thetasave = np.zeros((numchain,
                          sampperchain,
                          thetac.shape[1]))
    covmat0 = np.cov(thetac.T)
    if thetac.shape[1] > 1:
        covmat0 = 0.9*covmat0 + 0.1*np.diag(np.diag(covmat0))
        W, V = np.linalg.eigh(covmat0)
        hc = V @ np.diag(np.sqrt(W)) @ V.T
    else:
        hc = np.sqrt(covmat0)
        hc = hc.reshape(1,1)
        covmat0 = covmat0.reshape(1,1)
    tau = -1
    rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
    adjrho = rho*temps**(1/3)
    numtimes = 0
    for k in range(0, samptunning+sampperchain):
        rvalo = np.random.normal(0, 1, thetac.shape)
        rval = np.sqrt(2) * adjrho * (rvalo @ hc)
        thetap = thetac + rval
        if logpostf_grad is not None:
            diffval = (adjrho ** 2) * (dfval @ covmat0)
            thetap += diffval
            fvalp, dfvalp = logpostf(thetap)  # thetap : no chain x dimension
            fvalp = fvalp / temps  # to flatten the posterior
            dfvalp = dfvalp / temps
            term1 = rvalo / np.sqrt(2)
            term2 = (adjrho / 2) * ((dfval + dfvalp) @ hc)
            qadj = -(2 * np.sum(term1 * term2, 1) + np.sum(term2**2, 1))
        else:
            fvalp = logpostf_nograd(thetap)  # thetap : no chain x dimension
            fvalp = fvalp / temps
            qadj = np.zeros(fvalp.shape)
        swaprnd = np.log(np.random.uniform(size=fval.shape[0]))
        whereswap = np.where(np.squeeze(swaprnd)
                             < np.squeeze(fvalp - fval)
                             + np.squeeze(qadj))[0]
        if whereswap.shape[0] > 0:
            numtimes = numtimes + np.sum(whereswap > -1)/totnumchain
            thetac[whereswap, :] = 1*thetap[whereswap, :]
            fval[whereswap] = 1*fvalp[whereswap]
            if logpostf_grad is not None:
                dfval[whereswap, :] = 1*dfvalp[whereswap, :]
        fvaln = fval*temps
        orderprop = tempexchange(fvaln, temps, iters=5)
        fval = fvaln[orderprop] / temps
        thetac = thetac[orderprop, :]
        if logpostf_grad is not None:
            dfvaln = temps * dfval
            dfval = (1 / temps) * dfvaln[orderprop, :]
        if (k < samptunning) and (k % 10 == 0):  # if not done with tuning
            tau = tau + 1 / np.sqrt(1 + k/10) * \
                  ((numtimes / 10) - taracc)
            rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
            adjrho = rho*(temps**(1/3))
            numtimes = 0
        elif(k >= samptunning):  # if done with tuning
            thetasave[:, k-samptunning, :] = 1 * thetac[numtemps:, ]
    thetasave = np.reshape(thetasave, (-1, thetac.shape[1]))
    theta = thetasave[np.random.choice(range(0, thetasave.shape[0]),
                                       size=numsamp), :]
    sampler_info = {'theta': theta, 'logpost': logpostf_nograd(theta)}

    return sampler_info


def tempexchange(lpostf, temps, iters=1):
    order = np.arange(0, lpostf.shape[0])
    for k in range(0, iters):
        rtv = np.random.choice(range(1, lpostf.shape[0]), lpostf.shape[0])
        for rt in rtv:
            rhoh = (1/temps[rt-1] - 1 / temps[rt])
            if ((lpostf[order[rt]]-lpostf[order[rt - 1]]) * rhoh >
                    np.log(np.random.uniform(size=1))):
                temporder = order[rt - 1]
                order[rt-1] = 1*order[rt]
                order[rt] = 1 * temporder
    return order
