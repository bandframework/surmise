import numpy as np
from surmise.utilities import sampler


def fit(fitinfo, emu, x, y, args=None):
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
    def logpostfull(theta):

        logpost = thetaprior.lpdf(theta)
        inds = np.where(np.isfinite(logpost))[0]
        logpost[inds] += loglik(fitinfo, emu, theta[inds, :], y, x, args)
        return logpost

    # Call the sampler
    if 'theta0' not in args.keys():
        if 'sampler' in args.keys():
            if args['sampler'] == 'LMC':
                args['theta0'] = thetaprior.rnd(1000)
        else:
            args['theta0'] = thetaprior.rnd(1)
    if 'stepParam' not in args.keys():
        args['stepParam'] = np.std(thetaprior.rnd(1000), axis=0)
    sampler_obj = sampler(logpostfunc=logpostfull, options=args)
    theta = sampler_obj.sampler_info['theta']

    # Update fitinfo dict
    fitinfo['thetarnd'] = theta
    fitinfo['y'] = y
    fitinfo['x'] = x
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


def loglik(fitinfo, emu, theta, y, x, args):
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
            s0 = emucov[:, k, :].reshape((n, n))
            CovMat = s0 + np.diag(np.squeeze(obsvar))
        else:
            s0 = emucov[:, k].reshape((n, 1))
            CovMat = np.diag(np.squeeze(s0)) + np.diag(np.squeeze(obsvar))

        # Get the decomposition of covariance matrix
        CovMatEigS, CovMatEigW = np.linalg.eigh(CovMat)

        # Calculate residuals
        resid = m0 - y

        CovMatEigInv = CovMatEigW @ np.diag(1/CovMatEigS) @ CovMatEigW.T
        loglikelihood[k] = float(-0.5 * resid.T @ CovMatEigInv @ resid -
                                 0.5 * np.sum(np.log(CovMatEigS)))

    return loglikelihood
