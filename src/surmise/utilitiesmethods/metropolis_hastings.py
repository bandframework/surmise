import numpy as np
import scipy.stats as sps

"""Metropolis Hastings"""


def sampler(logpost_func,
            draw_func,
            scipy_stats_rng,
            specification):
    '''


    Parameters
    ----------
    logpost_func : function
        a function returns the log of the posterior for a given theta.
    draw_func : function
        a function returns random draws of initial design theta
    numsamp : int, optional
        number of samples to draw. The default is 2000.
    theta0 : array, optional
        initial theta value. The default is None.
    stepType : str, optional
        either 'uniform' or 'normal'. The default is 'normal'.
    stepParam : array, optional
        scaling parameter. The default is None.

    Returns
    -------
    sampler_info : dict
        returns numsamp random draws from posterior.

    '''
    # Hardcoded values
    VALID_SPECS = {
        "nSamples", "nBurnSamples", "theta0", "stepType", "stepParam", "verbose"
    }
    LOG_RATE = 25_000

    # Get specification values
    # TODO: Error check these with useful error messages
    assert set(specification) == VALID_SPECS
    numsamp = specification["nSamples"]
    burnSamples = specification["nBurnSamples"]
    theta0 = specification["theta0"]
    stepType = specification["stepType"]
    stepParam = specification["stepParam"]
    verbose = specification["verbose"]

    if verbose:
        # Don't log theta0 as it could potentially be an overwhelming amount of
        # information.  This could be an arugment for having logging at
        # different levels of detail.  Maybe a user needs to see the full theta0
        # in some cases.
        print(f"nSamples     = {numsamp}")
        print(f"nBurnSamples = {burnSamples}")
        print(f"stepType     = {stepType}")
        print(f"stepParam    = {stepParam}")

    # random number generator
    if not isinstance(scipy_stats_rng, np.random.Generator):
        raise TypeError("Given RNG is not a valid scipy.stats RNG")

    # scaling parameter
    if stepParam is None:
        stepParam = np.std(draw_func(burnSamples), axis=0)

    # For the current, symmetric step proposal distributions, we can create a
    # frozen step proposal distribution object up front and draw from it blindly
    # within the MCMC loop to determine the sample proposal with no need for
    # conditionals inside the loop.
    if stepType.lower() == "uniform":
        a, b = [-0.5, 0.5]
        length = b - a
        step_distribution = sps.uniform(loc=a, scale=length)
    elif stepType.lower() == "normal":
        mean, std = (0.0, 1.0)
        step_distribution = sps.norm(loc=mean, scale=std)
    else:
        raise ValueError("Bad step type {stepType}")

    # intial theta to start the chain
    if theta0 is None:
        theta0 = draw_func(1)

    p = theta0.shape[1]
    theta = np.full((burnSamples + numsamp, p), np.nan, float)
    theta[0] = theta0

    lposterior = np.full(burnSamples + numsamp, np.nan, float)
    lposterior[0] = np.squeeze(logpost_func(theta0, return_grad=False))
    if not np.isfinite(lposterior[0]):
        assert lposterior[0] == -np.inf
        raise RuntimeError("Initial theta evaluates to zero density")

    # We implicitly treat theta0 as accepted.  If the number of burn-in samples
    # is positive, we also treat it as part of the burn-in.
    n_acc = 1 if burnSamples == 0 else 0
    n_official_i = 1 if burnSamples == 0 else 0
    lposterior_list = []
    for i in range(1, burnSamples + numsamp):
        # Candidate theta
        step = step_distribution.rvs(size=p, random_state=scipy_stats_rng)
        theta_cand = theta[i-1, :] + stepParam * step
        if not all(np.isfinite(theta_cand)):
            raise RuntimeError("Proposed theta contains invalid values")
        theta_cand = np.reshape(np.array(theta_cand), (1, p))

        # Compute loglikelihood
        logpost = np.squeeze(logpost_func(theta_cand, return_grad=False))

        if logpost == -np.inf:
            accept = False
        elif not np.isfinite(logpost):
            raise ValueError(f"Invalid log posterior evaluation ({logpost})")
        elif logpost >= lposterior[i-1]:
            # Handle easy case directly, which precludes any possibility of an
            # overflow when exponentiating the difference in successive log
            # posterior values below.
            #
            # This also prevents unnecessary Bernoulli draws.
            accept = True
        else:
            # While analytically p_accept must be in (0, 1) here, if the
            # magnitude of the difference is large enough numerically this will
            # underflow to zero, which will sensibly result in the proposal
            # being rejected.
            #
            # In testing this, I found that np.exp(-745.0) = 5e-324, which
            # indicates the use of subnormal numbers before underflowing.
            p_accept = np.exp(logpost - lposterior[i-1])
            if p_accept == 0.0:
                accept = False
            else:
                assert 0.0 < p_accept < 1.0
                accept = (sps.bernoulli.rvs(p=p_accept, size=1, random_state=scipy_stats_rng) == 1)

        # Accept candidate?
        if accept:
            # Update position
            theta[i, :] = theta_cand
            lposterior[i] = logpost
            lposterior_list.append(logpost)
            if i >= burnSamples:
                n_acc += 1
        else:
            theta[i, :] = theta[i-1, :]
            lposterior[i] = lposterior[i-1]
            lposterior_list.append(logpost)

        # N official samples completed by end of i^th iteration
        # - Nonpositive value indicates still in warm-up phase
        n_official_i = i - burnSamples + 1

        if verbose:
            if (n_official_i >= 1) and (n_official_i % LOG_RATE == 0):
                acc_rate = n_acc / float(n_official_i)
                assert 0.0 <= acc_rate <= 1.0
                print(
                    f"Sample {n_official_i:>10} acceptance rate={acc_rate}"
                )
    assert n_official_i == numsamp
    acc_rate = n_acc / float(numsamp)
    assert 0.0 <= acc_rate <= 1.0

    theta = theta[(burnSamples):(burnSamples + numsamp), :]
    sampler_info = {'theta': theta, 'acc_rate': acc_rate,
                    'lpostlist': np.array(lposterior_list)}
    if verbose:
        print("Final Acceptance Rate: ", sampler_info["acc_rate"])
    return sampler_info
