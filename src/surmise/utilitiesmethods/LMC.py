import numpy as np
import scipy.optimize as spo

r'''
Metropolis-adjusted Langevin algorithm or Langevin Monte Carlo (LMC).

The LMC sampler is available through calling the `calibrator` object with an
optional argument `args={'sampler': 'LMC'}`.  LMC is a Markov chain Monte
Carlo method that seeks to propose the next iterates by leveraging gradient
information at the current iterate.  The proposal has the form

.. math::

    \theta^{k+1} = \theta^k - \nabla g(\theta^k) \Delta t +
    \sqrt{2\Delta t} Z,

where :math:`\Delta t` is a time stepsize, and :math:`Z` is an independently
and identically drawn sample from the standard Gaussian normal of the
appropriate dimension.  The proposal is then accepted or rejected by the
typical Metropolis-Hastings step, i.e. accept with probability

.. math::

    \alpha = \min\left\{1, \frac{\pi(\tilde{\theta}^{k+1})q(\theta^k \mid
    \tilde{\theta}^{k+1})}{\pi(\theta^{k})q(\tilde{\theta}^{k+1} \mid
    \theta^k)}\right\},

where :math:`\pi(\cdot)` is the posterior distribution, :math:`q(\cdot \mid
\cdot)` is the proposal distribution, and :math:`\theta^k,
\tilde{\theta}^{k+1}` are the current and the proposed point respectively.

Langevin Monte Carlo has shown strengths in increasing the acceptance rate,
compared to the typical Metropolis-Hastings algorithm (Roberts and
Rosenthal, 1998).  However, its significant drawback lies in its poor
scaling due to the computation for the gradient at the current iterate.

Refer to G. O. Roberts and J. S. Rosenthal. Optimal scaling of discrete
approximations to langevin diffusions. *Journal of the Royal Statistical
Society: Series B (Statistical Methodology)*, 60(1):255-268, 1998.
'''


def sampler(logpost_func,
            draw_func,
            numsamp=2000,
            theta0=None,
            **lmc_options):
    '''

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
    options : dict
        a dictionary contains the output of the sampler.
        Required -
            theta0: an m by p matrix of initial parameter values.
        Optional -
            numsamp: the number of samplers you want from the posterior.
            Default is 2000.

    Returns
    -------
    TYPE
        numsamp by p of sampled parameter values

    '''

    if theta0 is None:
        theta0 = draw_func(1000)

    # Minimum effective sample size (ESS) desired in the returned samples
    tarESS = np.max((150, 10 * theta0.shape[1]))

    # Test
    testout = logpost_func(theta0[0:2, :])
    if type(testout) is tuple:
        if len(testout) != 2:
            raise ValueError('log density does not return 1 or 2 elements')
        if testout[1].shape[1] is not theta0.shape[1]:
            raise ValueError('derivative appears to be the wrong shape')

        logpostf = logpost_func

        def logpostf_grad(theta):
            return logpost_func(theta)[1]

        try:
            testout = logpost_func(theta0[10, :], return_grad=False)
            if type(testout) is tuple:
                raise ValueError('Cannot stop returning a grad')

            def logpostf_nograd(theta):
                return logpost_func(theta, return_grad=False)

        except Exception:
            def logpostf_nograd(theta):
                return logpost_func(theta)[0]
    else:
        logpostf_grad = None
        logpostf = logpost_func
        logpostf_nograd = logpost_func

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
            raise ValueError('Could not find any points to vary.')

    thetaop = theta0[:10, :]
    thetastart = theta0
    thetac = np.mean(theta0, 0)
    thetas = np.maximum(np.std(theta0, 0), 10 ** (-8) * np.std(theta0))

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

    # begin preoptimizer
    for k in range(0, thetaop.shape[0]):
        theta0 = (thetaop[k, :] - thetac) / thetas
        if logpostf_grad is None:
            opval = spo.minimize(neglogpostf_nograd,
                                 theta0,
                                 method='L-BFGS-B',
                                 bounds=bounds)
            thetaop[k, :] = thetac + thetas * opval.x
        else:
            if keeptryingwithgrad:
                opval = spo.minimize(neglogpostf_nograd,
                                     theta0,
                                     method='L-BFGS-B',
                                     jac=neglogpostf_grad,
                                     bounds=bounds)
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
    # end Preoptimizer

    thetasave = np.vstack((thetastart, thetaop))
    Lsave = logpostf_nograd(thetasave)
    tau = -1
    rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
    numchain = 100
    maxiters = 10
    numsamppc = 200
    covmat0 = np.diag(thetas)
    for iters in range(0, maxiters):
        startingv = np.random.choice(np.arange(0, Lsave.shape[0]),
                                     size=Lsave.shape[0])
        thetasave = thetasave[startingv, :]

        covmat0 = 0.1*covmat0 + 0.9*np.cov(thetasave.T)

        if covmat0.ndim > 1:
            covmat0 += 0.1 * np.diag(np.diag(covmat0))
            Wc, Vc = np.linalg.eigh(covmat0)
            hc = (Vc @ np.diag(np.sqrt(Wc)) @ Vc.T)
        else:
            hc = np.sqrt(covmat0)

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
            whereswap = np.where(np.squeeze(swaprnd)
                                 < np.squeeze(fvalp - fval)
                                 + np.squeeze(qadj))[0]
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

        # termination criteria
        if iters > 1.5 and accr > taracc/2 and accr < 1.5*taracc and\
                (np.mean(ESS) > tarESS):
            break
        elif accr < taracc*4/5 or accr > taracc*5/4:
            tau = tau + 1/(0.2 + 0.2*iters) * (accr - taracc)
            rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
        if accr < taracc*1.5 and accr > taracc*0.6:
            trm = np.min((1.5*tarESS/np.mean(ESS), 4))
            numsamppc = np.ceil(numsamppc*trm).astype('int')

    theta = thetasave[np.random.choice(range(0, thetasave.shape[0]),
                                       size=numsamp), :]
    sampler_info = {'theta': theta, 'logpost': Lsave}

    return sampler_info
