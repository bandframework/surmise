import numpy as np

"""Adaptive Metropolis, Haario et al. 2001"""


def sampler(logpost_func,
            draw_func,
            numsamp=2000,
            theta0=None,
            Cov0=None,
            n0=30,
            stepScale=None,
            epsilonCov=1e-9,
            burnSamples=1000,
            verbose=True,
            **mh_options):
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
    Cov0 : array, optional
        initial proposal covariance value. The default is None.
    n0 : int, optional
        number of samples to use for covariance estimation.
    stepScale : float, optional
        multiplication constant for proposal covariance. The default is 2.4**2 / p, where p is
        the dimension of parameter (Gelman et al., 1996).
    epsilonCov : float, optional
        constant to avoid singular matrix. The default is 1e-9.
    **mh_options : dict
        additional options.

    Returns
    -------
    sampler_info : dict
        returns numsamp random draws from posterior.

    '''
    # intial theta to start the chain
    if theta0 is None:
        theta0 = draw_func(1)

    if Cov0 is None:
        simtheta = draw_func(n0)
        Cov0 = np.cov(simtheta.T)
        cumMean0 = np.mean(simtheta, 0)
    else:
        n0 = 1
        cumMean0 = theta0

    p = theta0.shape[1]
    # scaling parameter
    if stepScale is None:
        stepScale = 2.4 ** 2 / p

    lposterior = np.zeros(burnSamples + numsamp)
    theta = np.zeros((burnSamples + numsamp, theta0.shape[1]))
    # print(theta0)
    lposterior[0] = logpost_func(theta0, return_grad=False)
    theta[0, :] = theta0
    n_acc = 0

    # adaptive proposal covariance
    Covt = Cov0
    cumMeant = cumMean0

    lposterior_list = []

    for i in range(1, burnSamples + numsamp):
        if verbose:
            if i % 30000 == 0:
                print("At sample {}, acceptance rate is {}.".format(i, n_acc / i))
        # Candidate theta
        theta_cand = np.random.default_rng().multivariate_normal(mean=theta[i - 1],
                                                                 cov=Covt,
                                                                 size=1)

        theta_cand = np.reshape(np.array(theta_cand), (1, p))

        # Compute loglikelihood
        logpost = logpost_func(theta_cand, return_grad=False)

        if np.isfinite(logpost):
            p_accept = min(1, np.exp(logpost - lposterior[i - 1]))
            accept = np.random.uniform() < p_accept
        else:
            accept = False

        # Accept candidate?
        if accept:
            # Update position
            theta[i, :] = theta_cand
            lposterior[i] = logpost
            lposterior_list.append(logpost)
            if i >= burnSamples:
                n_acc += 1
        else:
            theta[i, :] = theta[i - 1, :]
            lposterior[i] = lposterior[i - 1]
            lposterior_list.append(logpost)

        # Update proposal covariance
        oldMeant = cumMeant
        cumMeant = (i + n0 - 1) / (i + n0) * cumMeant + 1 / (i + n0) * theta[i]
        Covt = (i + n0 - 1) / (i + n0) * Covt + (
                stepScale / (i + n0) * ((i + n0) * np.outer(oldMeant, oldMeant) -
                                        (i + n0 - 1) * np.outer(cumMeant, cumMeant) +
                                        np.outer(theta[i], theta[i]) +
                                        epsilonCov * np.eye(p))
        )

    theta = theta[(burnSamples):(burnSamples + numsamp), :]
    sampler_info = {'theta': theta, 'acc_rate': n_acc / numsamp,
                    'lpostlist': np.array(lposterior_list)}
    if verbose:
        print("Final Acceptance Rate: ", n_acc / numsamp)
    return sampler_info
