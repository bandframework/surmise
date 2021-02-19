import numpy as np
import scipy.stats as sps

"""Metropolis Hastings"""


def sampler(logpostfunc, options={}):
    '''
    Parameters
    ----------
    logpostfunc : function
        a function returns the log of the posterior for a given theta.
    options : dict, optional
        a dictionary providing the sampler options. The default is {}.
        The possible parameters for the sampler:

        - numsamp (int) : number of samples to draw

        - stepType : either 'uniform' or 'normal'

        - stepParam (float) : scaling parameter

        - theta0 : initial theta value

    Raises
    ------
    ValueError
        If a stepParam or theta0 is not provided.

    Returns
    -------
    sampler_info : dict
        a dictionary contains the output of the sampler.

    '''

    # Initialize
    if 'numsamp' in options.keys():
        n = options['numsamp']
    else:
        n = 2000

    if 'stepType' in options.keys():
        stepType = options['stepType']
    else:
        # default is normal
        stepType = 'normal'

    # scaling parameter
    if 'stepParam' in options.keys():
        stepParam = options['stepParam']
    else:
        raise ValueError('Unknown stepParam')

    # intial theta to start the chain
    if 'theta0' in options.keys():
        thetastart = options['theta0']
    else:
        raise ValueError('Unknown theta0')

    p = thetastart.shape[1]
    lposterior = np.zeros(1000 + n)
    theta = np.zeros((1000 + n, thetastart.shape[1]))
    lposterior[0] = logpostfunc(thetastart)
    theta[0, :] = thetastart
    n_acc = 0

    for i in range(1, 1000 + n):
        # Candidate theta
        if stepType == 'normal':
            theta_cand = [theta[i-1, :][k] + stepParam[k] *
                          sps.norm.rvs(0, 1, size=1) for k in range(p)]
        elif stepType == 'uniform':
            theta_cand = [theta[i-1, :][k] + stepParam[k] *
                          sps.uniform.rvs(-0.5, 0.5, size=1) for k in range(p)]

        theta_cand = np.reshape(np.array(theta_cand), (1, p))

        # Compute loglikelihood
        logpost = logpostfunc(theta_cand)

        if np.isfinite(logpost):
            p_accept = min(1, np.exp(logpost - lposterior[i-1]))
            accept = np.random.uniform() < p_accept
        else:
            accept = False

        # Accept candidate?
        if accept:
            # Update position
            theta[i, :] = theta_cand
            lposterior[i] = logpost
            if i >= n:
                n_acc += 1
        else:
            theta[i, :] = theta[i-1, :]
            lposterior[i] = lposterior[i-1]

    theta = theta[(1000):(1000 + n), :]
    sampler_info = {'theta': theta, 'acc_rate': n_acc/n}
    return sampler_info
