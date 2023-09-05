import numpy as np
import scipy.optimize as spo

'''
Parallel-Tempering Ensemble MCMC (uses Langevin Monte Carlo)
'''


def sampler(logpostfunc,
            draw_func,
            theta0=None,
            numsamp=2000,
            numtemps=32,
            numchain=16,
            sampperchain=400,
            maxtemp=30,
            **ptlmc_options):
    """

    Parameters
    ----------
    logpostfunc : function
        A function call describing the log of the posterior distribution.
            If no gradient, logpostfunc should take a value of an m by p numpy
            array of parameters and theta and return
            a length m numpy array of log posterior evaluations.
            If gradient, logpostfunc should return a tuple.  The first element
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
    **ptlmc_options : additional options
        This is a dictionary containing additional options a user might have passed but are not directly listed above.
        In general, we should not pass options this way.

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
    temps = np.array(temps, ndmin=2).T
    # number of optimization at each chain before starting
    numopt = temps.shape[0]
    # before beginning, let's test out the given logpdf function
    testout = logpostfunc(theta0[0:2, :])
    if type(testout) is tuple:
        if len(testout) != 2:
            raise ValueError('log density does not return 1 or 2 elements')
        if testout[1].shape[1] is not theta0.shape[1]:
            raise ValueError('derivative appears to be the wrong shape')
        logpostf = logpostfunc

        def logpostf_grad(thetain):
            return logpostfunc(thetain)[1]
        try:
            testout = logpostfunc(theta0[10, :], return_grad=False)
            if type(testout) is tuple:  # make sure that return_grad functionality works
                raise ValueError('Cannot stop returning a grad')

            def logpostf_nograd(theta):
                return logpostfunc(theta, return_grad=False)
        except Exception:
            def logpostf_nograd(theta):  # if not, do not use return_grad key
                return logpostfunc(theta)[0]
    else:
        logpostf_grad = None  # sometimes no derivative is given
        logpostf = logpostfunc
        logpostf_nograd = logpostfunc

    if logpostf_grad is None:  # these are standard parameters if there is
        taracc = 0.25  # close to theoretical result 0.234
    else:
        taracc = 0.60  # close to theoretical result in LMC paper
    # begin preoptimizer
    # order the existing initial theta's by log pdf
    ord1 = np.argsort(-np.squeeze(logpostf_nograd(theta0)) +
                      (theta0.shape[1] *
                       np.random.standard_normal(size=theta0.shape[0])**2))
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
            return -thetas * logpostf_grad(theta.reshape((1, len(theta))))
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
            r = (V.T*np.sqrt(W)) @ (V @ np.random.standard_normal(size=thetacen.shape[0]))

            if (neglogpostf_nograd((stepadj * r + opval.x)) -
                    l0) < 3*thetacen.shape[0]:
                thetaop[k, :] = thetacen + thetas * (stepadj * r + opval.x)
                notmoved = False
            else:
                stepadj /= 2
            if stepadj < 1/16:
                thetaop[k, :] = thetacen + thetas * opval.x
                notmoved = False
    # end preoptimizer
    # initialize the starting point
    thetac = thetaop
    if logpostf_grad is not None:
        fval, dfval = logpostf(thetac)
        fval = fval/temps
        dfval = dfval/temps
    else:
        fval = logpostf_nograd(thetac)
        fval = fval/temps
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
    numtimes = 0  # number of times we reject, just to star
    for k in range(0, samptunning+sampperchain):  # loop over all chains
        rvalo = np.random.normal(0, 1, thetac.shape)
        rval = np.sqrt(2) * adjrho * (rvalo @ hc)
        thetap = thetac + rval
        if logpostf_grad is not None:
            # calculate the elements to move if there is a gradiant
            diffval = (adjrho ** 2) * (dfval @ covmat0)
            thetap += diffval
            fvalp, dfvalp = logpostf(thetap)  # thetap : no chain x dimension
            fvalp = fvalp / temps  # to flatten the posterior
            dfvalp = dfvalp / temps
            term1 = rvalo / np.sqrt(2)
            term2 = (adjrho / 2) * ((dfval + dfvalp) @ hc)
            qadj = -(2 * np.sum(term1 * term2, 1) + np.sum(term2**2, 1))
        else:
            # calculate the elements to move if there is not a gradiant
            fvalp = logpostf_nograd(thetap)  # thetap : no chain x dimension
            fvalp = fvalp / temps
            qadj = np.zeros(fvalp.shape)
        swaprnd = np.log(np.random.uniform(size=fval.shape[0]))
        whereswap = np.where(np.squeeze(swaprnd)
                             < np.squeeze(fvalp - fval)
                             + np.squeeze(qadj))[0]  # MH step to find which of the chains to swap
        if whereswap.shape[0] > 0:  # if we swap, do it where needed
            numtimes = numtimes + np.sum(whereswap > -1)/totnumchain
            thetac[whereswap, :] = 1*thetap[whereswap, :]
            fval[whereswap] = 1*fvalp[whereswap]
            if logpostf_grad is not None:
                dfval[whereswap, :] = 1*dfvalp[whereswap, :]
        # do some swaps along the temperatures
        fvaln = fval*temps
        orderprop = tempexchange(fvaln, temps, iters=5)  # go through 5 times, swapping where needed
        fval = fvaln[orderprop] / temps
        thetac = thetac[orderprop, :]
        if logpostf_grad is not None:
            dfvaln = temps * dfval
            dfval = (1 / temps) * dfvaln[orderprop, :]
        # if we have to tune, let's move tau up or down which gives bigger or smaller jumps
        if (k < samptunning) and (k % 10 == 0):  # if not done with tuning
            tau = tau + 1 / np.sqrt(1 + k/10) * \
                  ((numtimes / 10) - taracc)
            rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
            adjrho = rho*(temps**(1/3))  # adjusting rho across the chain
            numtimes = 0
        elif k >= samptunning:  # if done with tuning
            thetasave[:, k-samptunning, :] = 1 * thetac[numtemps:, ]
    # save the theta values in the temp=1 chains, squeezing flattening the values of all chains
    thetasave = np.reshape(thetasave, (-1, thetac.shape[1]))
    # save random values from the chain of size numsamp
    theta = thetasave[np.random.choice(range(0, thetasave.shape[0]),
                                       size=numsamp), :]
    # store this in a dictionary
    sampler_info = {'theta': theta, 'logpost': logpostf_nograd(theta)}
    return sampler_info


def tempexchange(lpostf, temps, iters=1):
    # This function will swap values along the chain given the log pdf values in an
    # array lpostf with temperature array temps. It will do it iters number of times.
    # It returns the (random) revised order.
    order = np.arange(0, lpostf.shape[0])  # initializing
    for k in range(0, iters):
        rtv = np.random.choice(range(1, lpostf.shape[0]), lpostf.shape[0])  # choose random values to check for swapping
        for rt in rtv:
            rhoh = (1/temps[rt-1] - 1 / temps[rt])
            if ((lpostf[order[rt]]-lpostf[order[rt - 1]]) * rhoh >
                    np.log(np.random.uniform(size=1))):  # swap via the PT rule
                temporder = order[rt - 1]
                order[rt-1] = 1*order[rt]
                order[rt] = 1 * temporder
    return order
