import numpy as np
import scipy.stats as sps

"""Metropolis Hastings"""


def sampler(logpost_func,
            draw_func,
            numsamp=2000,
            theta0=None,
            stepType='normal',
            stepParam=None,
            burnSamples=1000,
            verbose=False,
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
    stepType : str, optional
        either 'uniform' or 'normal'. The default is 'normal'.
    stepParam : array, optional
        scaling parameter. The default is None.
    **mh_options : dict
        additional options.

    Returns
    -------
    sampler_info : dict
        returns numsamp random draws from posterior.

    '''

    # scaling parameter
    if stepParam is None:
        stepParam = np.std(draw_func(burnSamples), axis=0)

    # intial theta to start the chain
    if theta0 is None:
        theta0 = draw_func(1)

    p = theta0.shape[1]
    lposterior = np.zeros(burnSamples + numsamp)
    theta = np.zeros((burnSamples + numsamp, theta0.shape[1]))
    # print(theta0)
    lposterior[0] = logpost_func(theta0, return_grad=False).item()
    theta[0] = theta0
    n_acc = 0

    lposterior_list = []

    for i in range(1, burnSamples + numsamp):
        if verbose:
            if i % 30000 == 0:
                print("At sample {}, acceptance rate is {}.".format(i, n_acc/i))
        # Candidate theta
        theta_cand = None
        if stepType == 'normal':
            theta_cand = [theta[i-1, :][k] + stepParam[k] *
                          sps.norm.rvs(0, 1, size=1) for k in range(p)]
        elif stepType == 'uniform':
            theta_cand = [theta[i-1, :][k] + stepParam[k] *
                          sps.uniform.rvs(-0.5, 0.5, size=1) for k in range(p)]

        theta_cand = np.reshape(np.array(theta_cand), (1, p))

        # Compute loglikelihood
        logpost = logpost_func(theta_cand, return_grad=False).item()

        if np.isfinite(logpost):
            logp_accept = min(0, logpost - lposterior[i-1])
            # p_accept = min(1, np.exp(logpost - lposterior[i-1]))
            accept = np.random.uniform() < np.exp(logp_accept)
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
            theta[i, :] = theta[i-1, :]
            lposterior[i] = lposterior[i-1]
            lposterior_list.append(logpost)

    theta = theta[(burnSamples):(burnSamples + numsamp), :]
    sampler_info = {'theta': theta, 'acc_rate': n_acc/numsamp,
                    'lpostlist': np.array(lposterior_list)}
    if verbose:
        print("Final Acceptance Rate: ", n_acc/numsamp)
    return sampler_info
