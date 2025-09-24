import numpy as np
from surmise.utilities import sampler
import copy


def fit(fitinfo, emu, x, y, **bayes_args):
    '''
    The main required function to be called by calibration to fit a
    calibration model.

    .. note::
        This approach uses Bayesian posterior sampling using the following
        steps:

            - 1. Take the emulator to approximate the computer model
              simulations
            - 2. Obtain the emulator predictive mean values at a given theta
              and x
            - 3. Calculate the residuals between emulator predictions and
              observed data
            - 4. Provide the log-posterior as the sum of log-prior and
              log-likelihood
            - 5. Use Monte Carlo or nested sampling method to sample the
              posterior

    Parameters
    ----------
    fitinfo : dict
        A dictionary including the calibration fitting information once
        complete.
        The dictionary is passed by reference, so it returns None.
        Note that the following are preloaded:

        - fitinfo['thetaprior'].rnd(s) : Get s random draws from the prior
          predictive distribution on theta.

        - fitinfo['thetaprior'].lpdf(theta) : Get the logpdf at theta(s).

        In addition, calibration can directly use:

        - fitinfo['thetamean'] : the mean of the prediction of theta

        - fitinfo['thetavar'] : the var of the predictive variance on theta

        - fitinfo['thetarand'] : some number draws from the predictive
          distribution on theta

    emu : :class: `surmise.emulation.emulator`
        An emulator class instance as defined in emulation
        Example emu functions
        (Not all of these will work, it depends on the emulation software.)

        - emupredict = emu(theta, x).predict()

        - emupredict.mean() : an array of size (theta.shape[0], x.shape[0])
          containing the mean of the target function at theta and x

        - emupredict.var() : an array of size (theta.shape[0], x.shape[0])
          containing the variance of the target function at theta and x

        - emupredict.cov() : an array of size
          (theta.shape[0], x.shape[0], x.shape[0]) containing the
          covariance matrix in x at each theta.

        - emupredict.rand(s) : an array of size
          (s, theta.shape[0], x.shape[0]) containing s random draws from
          the emulator at theta and x.

    x : numpy.ndarray
        An array of x that represents the inputs.

    y : numpy.ndarray
        A one dimensional array of observed values at x.

    args : dict
        A dictionary containing options passed to the calibrator.

    '''
    thetaprior = fitinfo['thetaprior']

    # Define the posterior function
    def logpostfull(theta, return_grad=False):
        logpost = thetaprior.lpdf(theta)
        inds = np.where(np.isfinite(logpost))[0]
        if len(inds) > 0:
            logpost[inds] += loglik(fitinfo, emu, theta[inds, :], y, x)
        return logpost

    # Define the draw function to sample from initial theta
    def draw_func(n):
        p = thetaprior.rnd(1).shape[1]
        theta0 = np.array([]).reshape(0, p)

        if 'thetarnd' in fitinfo:
            theta0 = fitinfo['thetarnd']
        if '_emulator__theta' in dir(emu):
            theta0 = np.vstack((theta0, copy.copy(emu._emulator__theta)))
        n0 = len(theta0)
        if n0 < n:
            theta0 = np.vstack((thetaprior.rnd(n - n0), theta0))
        else:
            theta0 = theta0[np.random.randint(theta0.shape[0], size=n), :]

        return theta0

    # Call the sampler
    if 'sampler' in bayes_args.keys():
        name = bayes_args['sampler']
    else:
        name = 'unspecified'
    _ = name  # to satisfy flake8, can be removed when variable is used

    sampler_obj = sampler(logpost_func=logpostfull,
                          draw_func=draw_func,
                          **bayes_args)

    theta = sampler_obj.sampler_info['theta']

    # Update fitinfo dict
    fitinfo['thetarnd'] = theta
    fitinfo['y'] = y
    fitinfo['x'] = x
    fitinfo['emu'] = emu
    return


def thetarnd(fitinfo, s=100, args=None):
    '''
    Return s draws from the predictive distribution of theta (not required)

    Parameters
    ----------
    fitinfo : dict
        A dictionary including the calibration fitting information once
        complete.
    s : int, optional
        Size of the random draws. The default is 100.

    Returns
    -------
    numpy.ndarray
        s draws from the predictive distribution of theta.

    '''

    return fitinfo['thetarnd'][np.random.choice(fitinfo['thetarnd'].shape[0],
                                                size=s), :]


def loglik(fitinfo, emu, theta, y, x):
    '''

    Parameters
    ----------
    fitinfo : dict
        A dictionary including the calibration fitting information once
        complete.
        The dictionary is passed by reference, so there is no reason to
        return anything.
        Note that the following are preloaded:

        - fitinfo['thetaprior'].rnd(s) : Get s random draws from the prior
          predictive distribution on theta.

        - fitinfo['thetaprior'].lpdf(theta) : Get the logpdf at theta(s).

        In addition, calibration can directly use:

        - fitinfo['thetamean'] : the mean of the prediction of theta

        - fitinfo['thetavar'] : the var of the predictive variance on theta

        - fitinfo['thetarand'] : some number draws from the predictive
          distribution on theta

    emu : :class: `surmise.emulation.emulator`
        An emulator class instance as defined in emulation
        Example emu functions
        (Not all of these will work, it depends on the emulation software.)

        - emupredict = emu(theta, x).predict()

        - emupredict.mean() : an array of size (theta.shape[0], x.shape[0])
          containing the mean of the target function at theta and x

        - emupredict.var() : an array of size (theta.shape[0], x.shape[0])
          containing the variance of the target function at theta and x

        - emupredict.cov() : an array of size
          (theta.shape[0], x.shape[0], x.shape[0]) containing the
          covariance matrix in x at each theta.

        - emupredict.rand(s) : an array of size
          (s, theta.shape[0], x.shape[0]) containing s random draws from
          the emulator at theta and x.

    x : numpy.ndarray
        An array of x that represents the inputs.

    y : numpy.ndarray
        A one dimensional array of observed values at x.

    args : dict
        A dictionary containing options passed to the calibrator.

    '''

    if 'yvar' in fitinfo.keys():
        obsvar = fitinfo['yvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')

    # Obtain emulator results
    emupredict = emu.predict(x, theta)
    emumean = emupredict.mean()

    try:
        emucov = emupredict.covx()
        is_cov = True
    except Exception:
        emucov = emupredict.var()
        is_cov = False

    p = emumean.shape[1]
    n = emumean.shape[0]
    y = y.reshape((n, 1))

    loglikelihood = np.zeros((p, 1))

    for k in range(0, p):
        m0 = emumean[:, k].reshape((n, 1))

        # Compute the covariance matrix
        if is_cov is True:
            s0 = emucov[k, :, :].reshape((n, n))
            CovMat = s0 + np.diag(np.squeeze(obsvar))
        else:
            s0 = emucov[:, k].reshape((n, 1))
            CovMat = np.diag(np.squeeze(s0)) + np.diag(np.squeeze(obsvar))

        # Get the decomposition of covariance matrix
        CovMatEigS, CovMatEigW = np.linalg.eigh(CovMat)

        # Calculate residuals
        resid = m0 - y

        CovMatEigInv = CovMatEigW @ np.diag(1 / CovMatEigS) @ CovMatEigW.T
        loglikelihood[k] = (-0.5 * resid.T @ CovMatEigInv @ resid -
                            0.5 * np.sum(np.log(CovMatEigS))).item()

    return loglikelihood


def thetalpdf(info, theta, args=None):
    '''
    Returns log of the posterior of the given theta.

    Not required.
    '''
    emu = info['emu']
    y = info['y']
    x = info['x']
    thetaprior = info['thetaprior']
    logpost = thetaprior.lpdf(theta)
    if logpost.ndim > 0.5 and logpost.shape[0] > 1.5:
        inds = np.where(np.isfinite(logpost))[0]
        logpost[inds] += loglik(info, emu, theta[inds], y, x)
    elif np.isfinite(logpost):
        logpost += loglik(info, emu, theta, y, x)
    return logpost
