import ptemcee
import numpy as np

'''
Surmise wrapper ptemcee (https://github.com/willvousden/ptemcee)
'''


def sampler(logpost_func,
            draw_func,
            log_likelihood,
            log_prior,
            nburnin=100,
            ndim=15,
            niterations=200,
            ntemps=50,
            nthin=1,
            nwalkers=200,
            nthreads=10,
            Tmax=np.inf,
            verbose=False,
            **ptmc_options):
    """

    Parameters
    ----------
    logpostfunc: function,
        Not used in PTMC sampler. It uses log_likelihood and log_prior instead.

    draw_func: function, required
        A function that produces approximate draws from the prior distribution.
        This is used to initialize MCMC chains.
    log_likelihood: function, required
        Log of the likelihood.
    log_prior: function, required
        Log of the prior.
    nburnin:
        Number of burnin samples.
    ndim:
        Dimension of the model parameter space.
    niterations:
        Number of MCMC samples for each chain after burnin.
    nthin:
        Thinning applied to MCMC chains. The default is 1, which is no thinning.
    nwalkers:
        Number of chains.
    nthreads:
        Number of threads for parallel computation.
    ntemps: integer, optional
        A positive integer that controls how many chains of varying temperature to run simultaneously.
        The default is 50.
    Tmax: double, optional
        A number larger than 1 that gives the maximum temperature used in parallel tempering.
        The default is inf.
    verbose: bool, optional
        Boolean flag to control output printing.  The default is False (do not print).
    **ptlmc_options: additional options
        This is a dictionary containing additional options a user might have passed but are not directly listed above.
        In general, we should not pass options this way.

    Raises
    ------
    ValueError
        Indicates that something was not entered right, please check the documentation.

    Returns
    -------
    dictionary
        A dictionary that contains the sampled values in the key 'theta' and the corresponding log pdf values in the
        key 'logpost'.

    """
    nburnin = int(nburnin)
    ndim = int(ndim)
    niterations = int(niterations)
    ntemps = int(ntemps)
    nthin = int(nthin)
    nwalkers = int(nwalkers)
    nthreads = int(nthreads)
    Tmax = float(Tmax)
    global log_like
    def log_like(x): return log_likelihood(x.reshape(-1, ndim))
    global log_prior_fix
    def log_prior_fix(x): return log_prior(x.reshape(-1, ndim))
    # sampler = PTSampler(ntemps, nwalkers, ndim, logl, logp, threads=nthreads, betas=betas)
    ptsampler_ex = ptemcee.Sampler(nwalkers, ndim, log_like, log_prior_fix, ntemps, threads=nthreads, Tmax=Tmax)

    pos0 = np.array([draw_func(nwalkers) for n in range(0, ntemps)])
    if verbose:
        print("Running burn-in phase")
    for p, lnprob, lnlike in ptsampler_ex.sample(pos0, iterations=nburnin, adapt=True):
        pass
    ptsampler_ex.reset()  # Discard previous samples from the chain, but keep the position

    if verbose:
        print("Running MCMC chains")
    # 5. Now we sample for nwalkers*niterations, recording every nthin-th sample
    for p, lnprob, lnlike in ptsampler_ex.sample(p, iterations=niterations, thin=nthin, adapt=True):
        pass

    if verbose:
        print('Done MCMC')

    mean_acc_frac = np.mean(ptsampler_ex.acceptance_fraction)

    if verbose:
        print(f"Mean acceptance fraction: {mean_acc_frac:.3f}",
              f"(in total {nwalkers*niterations} steps)")

    # We only analyze the zero temperature MCMC samples

    samples = ptsampler_ex.chain[0, :, :, :].reshape((-1, ndim))

    sampler_info = {'theta': samples, 'acc_rate': mean_acc_frac}
    return sampler_info
