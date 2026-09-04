import numpy as np
import scipy.stats as sps
import scipy.optimize as spo

'''
Parallel-Tempering Ensemble MCMC (uses Langevin Monte Carlo)
'''


def sampler(logpost_func,
            draw_func,
            scipy_stats_rng,
            theta0=None,
            numsamp=2000,
            numtemps=32,
            numchain=16,
            sampperchain=400,
            maxtemp=30):
    """

    Parameters
    ----------
    logpost_func : function
        A function call describing the log of the posterior distribution.
            If no gradient, logpost_func should take a value of an m by p numpy
            array of parameters and theta and return
            a length m numpy array of log posterior evaluations.
            If gradient, logpost_func should return a tuple.  The first element
            in the tuple should be as listed above.
            The second element in the tuple should be an m by p matrix of
            gradients of the log posterior.
    draw_func : function, required
        A function that produces approximate draws from the distribution.  Can be used to initialize points.
    theta0 : n by p numpy array, optional
         This should contain a long list of original parameters to start from. The default is None.
    numsamp : integer, optional
        Number of samples returned from the posterior. The default is 2000.
    numtemps : integer, optional
        A positive integer that controls how many chains of varying temperature to run simultaneously. The default is
         32.
    numchain : integer, optional
        A positive integer that controls how many chains of fixed temperature to run simultaneously. The default is 16.
    sampperchain : integer, optional
        A positive integer that controls how many samples should be done for each chain. The default is 400.
    maxtemp : double, optional
        A positive number, larger than 1, that gives the maximum temperature used in parallel tempering. The default
        is 30.

    Raises
    ------
    ValueError
        Indicates that something was not entered right, please check documentation.

    Returns
    -------
    dictionary
        A dictionary that contains the sampled values in the key 'theta' and the corresponding log pdf values in the
        key 'logpost'.

    """

    # random number generator
    if not isinstance(scipy_stats_rng, np.random.Generator):
        raise TypeError("Given RNG is not a valid scipy.stats RNG")

    # If we do not get parameters to start, draw 1000
    if theta0 is None:
        theta0 = draw_func(1000)
    # Need to make sure the initial draws are sufficent to continue
    if theta0.shape[0] < 10*theta0.shape[1]:
        theta0 = draw_func(1000)
    # Setting up some default parameters
    fractunning = 0.5  # number of samples spent tunning the sampler
    # define the number of samples for tunning
    samptunning = np.ceil(sampperchain*fractunning).astype('int')
    # defining the total number of chains
    totnumchain = numtemps+numchain
    # spacing out the temperature vector to go from maxtemp to 1, and  then replacating 1 the number of
    # non-temperatured chains
    temps = np.concatenate((np.exp(np.linspace(np.log(maxtemp),
                                               np.log(maxtemp)/(numtemps+1),
                                               numtemps)),
                            np.ones(numchain)))  # ratio idea tend from emcee
    tempsc = temps[:, np.newaxis]  # for broadcasting against (chain, p) arrays

    # number of optimization at each chain before starting
    numopt = temps.shape[0]
    # before beginning, let's test out the given logpdf function
    testout = logpost_func(theta0[0:2, :])
    if type(testout) is tuple:
        if len(testout) > 2:
            raise ValueError('log density does not return 1 or 2 elements')
        if testout[1].shape[1] != theta0.shape[1]:
            raise ValueError('derivative appears to be the wrong shape')

        def logpostf(thetain):  # canonical shapes: (m,) and (m, p)
            f, df = logpost_func(thetain)
            f = np.asarray(f, dtype=float).ravel()
            df = np.asarray(df, dtype=float).reshape(f.shape[0], -1)
            return f, df

        def logpostf_grad(thetain):
            return logpostf(thetain)[1]
        try:
            testout = logpost_func(theta0[10, :], return_grad=False)
            if type(testout) is tuple:  # make sure that return_grad functionality works
                raise ValueError('Cannot stop returning a grad')

            def logpostf_nograd(theta):
                return np.asarray(logpost_func(theta, return_grad=False),
                                  dtype=float).ravel()
        except Exception:
            def logpostf_nograd(theta):  # if not, do not use return_grad key
                return np.asarray(logpost_func(theta)[0], dtype=float).ravel()
    else:
        logpostf_grad = None  # sometimes no derivative is given

        def logpostf_nograd(theta):
            return np.asarray(logpost_func(theta), dtype=float).ravel()
        logpostf = logpostf_nograd

    if logpostf_grad is None:  # these are standard parameters if there is
        taracc = 0.25  # close to theoretical result 0.234
    else:
        taracc = 0.60  # close to theoretical result in LMC paper
    # begin preoptimizer
    # order the existing initial theta's by log pdf
    ord1 = np.argsort(-logpostf_nograd(theta0) +
                      (theta0.shape[1] *
                       sps.norm.rvs(size=theta0.shape[0],
                                    random_state=scipy_stats_rng)**2))
    theta0 = theta0[ord1[0:totnumchain], :]
    # begin optimizing at each chain
    thetacen = np.mean(theta0, 0)
    thetas = np.maximum(np.std(theta0, 0), 10 ** (-8) * np.std(theta0))

    # rescale the input to make it easier to optimize
    def neglogpostf_nograd(thetap):
        theta = thetacen + thetas * thetap
        return -logpostf_nograd(theta.reshape((1, len(theta))))[0]
    if logpostf_grad is not None:
        def neglogpostf_grad(thetap):
            theta = thetacen + thetas * thetap
            return -thetas * logpostf_grad(theta.reshape((1, len(theta)))).ravel()
    boundL = np.maximum(-10*np.ones(theta0.shape[1]),
                        np.min((theta0 - thetacen)/thetas, 0))
    boundU = np.minimum(10*np.ones(theta0.shape[1]),
                        np.max((theta0 - thetacen)/thetas, 0))
    bounds = spo.Bounds(boundL, boundU)
    thetaop = theta0
    # now we are ready to optimize for each chain
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
        # use these as starting locations
        # try to move off optimized value to stop it from devolving
        W, V = np.linalg.eigh(opval.hess_inv @ np.eye(thetacen.shape[0]))
        notmoved = True
        if k == 0:
            notmoved = False
        stepadj = 4
        l0 = neglogpostf_nograd(opval.x)
        while notmoved:
            if (W > 0).all():
                r = (V.T*np.sqrt(W)) @ (V @ sps.norm.rvs(size=thetacen.shape[0],
                                                         random_state=scipy_stats_rng))
            else:
                stepadj /= 2
                if stepadj < 1/16:
                    thetaop[k, :] = thetacen + thetas * opval.x
                    notmoved = False
                continue

            if (neglogpostf_nograd(stepadj * r + opval.x) -
                    l0) < 3*thetacen.shape[0]:
                thetaop[k, :] = thetacen + thetas * (stepadj * r + opval.x)
                notmoved = False
            else:
                stepadj /= 2
    # end preoptimizer
    # initialize the starting point
    thetac = thetaop
    if logpostf_grad is not None:
        fval, dfval = logpostf(thetac)
        fval = fval / temps
        dfval = dfval / tempsc
    else:
        fval = logpostf_nograd(thetac) / temps

    # preallocate the saving matrix
    thetasave = np.zeros((numchain,
                          sampperchain,
                          thetac.shape[1]))
    # try to start the covariance matrix
    covmat0 = np.cov(thetac.T)
    if thetac.shape[1] > 1:
        covmat0 = 0.9*covmat0 + 0.1*np.diag(np.diag(covmat0))  # add a diagonal part to prevent any non-moving issues
        W, V = np.linalg.eigh(covmat0)
        hc = V @ np.diag(np.sqrt(W)) @ V.T
    else:
        hc = np.sqrt(covmat0)
        hc = hc.reshape(1, 1)
        covmat0 = covmat0.reshape(1, 1)
    # Parameter initilzation
    tau = -1
    rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
    adjrho = rho*temps**(1/3)  # this adjusts rho across different temperatures
    adjrhoc = adjrho[:, np.newaxis]
    numtimes = 0  # number of times we reject, just to star
    for k in range(0, samptunning+sampperchain):  # loop over all chains
        rvalo = sps.norm.rvs(size=thetac.shape, random_state=scipy_stats_rng)
        rval = (np.sqrt(2) * adjrho * np.squeeze(rvalo @ hc).T).T
        if thetac.shape[1] > 1:
            thetap = thetac + rval
        elif thetac.shape[1] == 1:
            thetap = thetac + rval[:, np.newaxis]
        if logpostf_grad is not None:
            # calculate the elements to move if there is a gradiant
            diffval = (adjrhoc ** 2) * (dfval @ covmat0)
            thetap += diffval
            fvalp, dfvalp = logpostf(thetap)  # thetap : no chain x dimension
            fvalp = fvalp / temps  # to flatten the posterior
            dfvalp = dfvalp / tempsc
            term1 = rvalo / np.sqrt(2)
            term2 = (adjrhoc / 2) * ((dfval + dfvalp) @ hc)
            qadj = -(2 * np.sum(term1 * term2, 1) + np.sum(term2**2, 1))
        else:
            # calculate the elements to move if there is not a gradiant
            fvalp = logpostf_nograd(thetap) / temps  # thetap : no chain x dimension
            qadj = np.zeros(fvalp.shape)
        swaprnd = np.log(sps.uniform.rvs(size=fval.shape[0], random_state=scipy_stats_rng))
        whereswap = np.where(np.squeeze(swaprnd)
                             < np.squeeze(fvalp - fval)
                             + np.squeeze(qadj))[0]  # MH step to find which of the chains to swap
        if whereswap.shape[0] > 0:  # if we swap, do it where needed
            numtimes = numtimes + np.sum(whereswap > -1)/totnumchain
            thetac[whereswap] = np.copy(thetap[whereswap])
            fval[whereswap] = np.copy(fvalp[whereswap])
            if logpostf_grad is not None:
                dfval[whereswap] = np.copy(dfvalp[whereswap])
        # do some swaps along the temperatures
        fvaln = fval * temps
        # go through 5 times, swapping where needed
        orderprop = tempexchange(fvaln, temps, iters=5, scipy_stats_rng=scipy_stats_rng)
        fval = fvaln[orderprop] / temps
        thetac = thetac[orderprop, :]
        if logpostf_grad is not None:
            dfvaln = tempsc * dfval
            dfval = (1 / tempsc) * dfvaln[orderprop, :]
        # if we have to tune, let's move tau up or down which gives bigger or smaller jumps
        if (k < samptunning) and (k % 10 == 0):  # if not done with tuning
            tau = tau + 1 / np.sqrt(1 + k/10) * \
                  ((numtimes / 10) - taracc)
            rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
            adjrho = rho*(temps**(1/3))  # adjusting rho across the chain
            adjrhoc = adjrho[:, np.newaxis]
            numtimes = 0
        elif k >= samptunning:  # if done with tuning
            thetasave[:, k-samptunning, :] = 1 * thetac[numtemps:, ]
    # save the theta values in the temp=1 chains, squeezing flattening the values of all chains
    thetasave_flatten = np.reshape(thetasave, (-1, thetac.shape[1]))
    # save random values from the chain of size numsamp
    # TODO: choose the first numsamp as required samples, the flattening should be revisited.
    theta = thetasave_flatten[:numsamp].copy()  # copy: do not alias 'theta_from_chain'
    # store this in a dictionary
    sampler_info = {'theta': theta, 'theta_from_chain': thetasave, 'logpost': logpostf_nograd(theta)}
    return sampler_info


def tempexchange(lpostf, temps, iters=1, scipy_stats_rng=None):
    # This function will swap values along the chain given the log pdf values in an
    # array lpostf with temperature array temps. It will do it iters number of times.
    # It returns the (random) revised order.
    assert scipy_stats_rng is not None

    order = np.arange(0, lpostf.shape[0])  # initializing
    for k in range(0, iters):
        # choose random values to check for swapping
        rtv = scipy_stats_rng.choice(range(1, lpostf.shape[0]), lpostf.shape[0])
        for rt in rtv:
            rhoh = (1/temps[rt-1] - 1 / temps[rt])
            if ((lpostf[order[rt]]-lpostf[order[rt - 1]]) * rhoh >
                    np.log(sps.uniform.rvs(size=1, random_state=scipy_stats_rng))):  # swap via the PT rule
                temporder = order[rt - 1]
                order[rt-1] = 1*order[rt]
                order[rt] = 1 * temporder
    return order
