import numpy as np
import scipy.stats as sps
from surmise.utilities import sampler
import copy


def fit(fitinfo, emu, x, y, **myargs):
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
        An arbitary dictionary where the fitting information is placed once
        complete.
        This dictionary is pass by reference, so there is no reason to return
        anything. Keep only stuff that will be used by predict below.

        Note that the following are preloaded:

        - fitinfo['thetaprior'].rnd(s) : Get s random draws from the prior
          predictive distribution on theta.

        - fitinfo['thetaprior'].lpdf(theta) : Get the logpdf at theta(s).

        The following are optional preloads based on user input:

        - fitinfo[yvar] : The vector of observation variances at y

        In addition, calibration can directly use and communicate back to the
        user if you include:

        - fitinfo['thetamean'] : the mean of the prediction of theta.

        - fitinfo['thetavar'] : the predictive variance of theta.

        - fitinfo['thetarnd'] : some number draws from the predictive
          distribution of theta.

        - fitinfo['lpdf'] :log of the posterior of the given theta.

    emu : surmise.emulation.emulator
        An emulator class instance as defined in emulation.
    x : numpy.ndarray
        An array of x  that represent the inputs.
    y : numpy.ndarray
        A one dimensional array of observed values at x.
    myargs : dict, optional
        A dictionary containing additional options passed. The default is None.

    Returns
    -------
    None.

    '''

    thetaprior = fitinfo['thetaprior']
    theta = thetaprior.rnd(10)
    emupredict = emu.predict(x, theta, args={'return_grad': True,
                                             'return_covx': False})
    try:
        lpdf, lpdf_grad = emupredict.lpdf(y,
                                          args={'addvar': fitinfo['yvar'],
                                                'return_grad': True})
    except ValueError:
        raise ValueError('need logpdf with return_grad=True')

    if 'lpdf_grad' not in dir(thetaprior):
        # if an emulator returns a gradient
        def lpdf_grad(theta):
            f_base = thetaprior.lpdf(theta)
            inds = np.where(np.isfinite(f_base))[0]
            grad = np.zeros((theta.shape[0], theta.shape[1]))
            n_finite = len(inds)
            for k in range(0, theta.shape[1]):
                thetaprop = copy.copy(theta)
                thetaprop[:, k] += 10**(-6)
                f_base2 = thetaprior.lpdf(thetaprop[inds, :])
                grad[inds, k] = 10**(6) * (f_base2 -
                                           f_base[inds]).reshape(n_finite,)

            return grad
        thetaprior.lpdf_grad = lpdf_grad

    def logpostfull_wgrad(theta, return_grad=True):
        logpost = thetaprior.lpdf(theta)
        inds = np.where(np.isfinite(logpost))[0]
        logpost = logpost.reshape(-1, 1)

        if return_grad:
            # obtain the gradient of the log-prior
            dlogpost = thetaprior.lpdf_grad(theta)
            # obtain the log-likelihood and the gradient of it
            emupredict = emu.predict(x, theta[inds, :],
                                     args={'return_grad': True,
                                           'return_covx': False})
            loglikinds, dloglikinds = emupredict.lpdf(y,
                                                      args={'addvar': fitinfo['yvar'],
                                                            'return_grad': True})
            logpost[inds] += loglikinds
            dlogpost[inds] += dloglikinds
            return logpost, dlogpost
        else:
            # obtain the log-likelihood
            emupredict = emu.predict(x, theta[inds, :],
                                     args={'return_grad': False,
                                           'return_covx': False})
            logpost[inds] += emupredict.lpdf(y,
                                             args={'addvar': fitinfo['yvar'],
                                                   'return_grad': False})
            return logpost

    def draw_func(n):
        p = thetaprior.rnd(1).shape[1]
        theta0 = np.array([]).reshape(0, p)

        if 'thetarnd' in fitinfo:
            theta0 = fitinfo['thetarnd']
        if '_emulator__theta' in dir(emu):
            theta0 = np.vstack((theta0, copy.copy(emu._emulator__theta)))
        n0 = len(theta0)
        if n0 < n:
            theta0 = np.vstack((thetaprior.rnd(n-n0), theta0))
        else:
            theta0 = theta0[np.random.randint(theta0.shape[0], size=n), :]

        return theta0
    sampler_obj = sampler(logpost_func=logpostfull_wgrad,
                          draw_func=draw_func,
                          **myargs)
    theta = sampler_obj.sampler_info['theta']
    # obtain log-posterior of theta values
    ladj = logpostfull_wgrad(theta, return_grad=False)
    mladj = np.max(ladj)
    fitinfo['lpdfapproxnorm'] = np.log(np.mean(np.exp(ladj - mladj))) + mladj
    fitinfo['thetarnd'] = theta
    fitinfo['y'] = y
    fitinfo['x'] = x
    fitinfo['emu'] = emu
    return


def predict(predinfo, fitinfo, emu, x, args=None):
    '''
    Finds prediction at x given the emulator and dictionary fitinfo.

    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where the prediction information is placed
        once complete. Key elements:

            - predinfo['mean'] : the mean of the prediction
            - predinfo['var'] : the variance of the prediction
            - predinfo['rand'] : random draws from the predictive distribution
              of theta.

    fitinfo : dict
        A dictionary including the calibration fitting information once
        complete.
    emu : surmise.emulation.emulator
        DESCRIPTION.
    x : TYPE
        An array of x values where the prediction occurs.
    args : dict, optional
        A dictionary containing options. The default is None.

    Returns
    -------
    None.

    '''

    theta = fitinfo['thetarnd']
    if theta.ndim == 1 and fitinfo['theta'].shape[1] > 1.5:
        theta = theta.reshape((1, theta.shape[0]))

    emupredict = emu.predict(x, theta)
    predinfo['rnd'] = copy.deepcopy(emupredict()).T

    emupredict = emu.predict(x, theta)
    emumean = emupredict.mean()
    emucovxhalf = emupredict.covxhalf()
    varfull = np.sum(np.square(emucovxhalf), axis=2)

    for k in range(0, theta.shape[0]):
        re = emucovxhalf[:, k, :] @ \
            sps.norm.rvs(0, 1, size=(emucovxhalf.shape[2]))
        predinfo['rnd'][k, :] += re

    predinfo['mean'] = np.mean(emumean, 1)
    predinfo['var'] = np.mean(varfull, 1) + np.var(emumean, 1)
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
