import numpy as np
import scipy.stats as sps
from surmise.utilities import sampler
import copy


def fit(fitinfo,
        emu,
        x,
        y,
        clf_method=None,
        myusedir=True,
        **bayeswoodbury_args):
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
    args : dict, optional
        A dictionary containing options passed. The default is None.

    Returns
    -------
    None.

    '''
    if clf_method is not None:
        raise NotImplementedError(
            "CLF method use is not under test nor officially offered yet"
        )

    thetaprior = fitinfo['thetaprior']

    if myusedir:
        try:
            theta = thetaprior.rnd(10)
            emupredict = emu.predict(x, theta, args={'return_grad': True})
            emupredict.mean_gradtheta()
            emureturn_grad = True
        except Exception:
            emureturn_grad = False
    else:
        emureturn_grad = False

    if emureturn_grad and 'lpdf_grad' not in dir(thetaprior):
        # if an emulator returns a gradient
        def lpdf_grad(theta):
            f_base = thetaprior.lpdf(theta)
            inds = np.where(np.isfinite(f_base))[0]
            grad = np.zeros((theta.shape[0], theta.shape[1]))
            n_finite = len(inds)

            if n_finite > 0:
                for k in range(0, theta.shape[1]):
                    thetaprop = copy.copy(theta)
                    thetaprop[:, k] += 10**(-3)
                    f_base2 = thetaprior.lpdf(thetaprop[inds, :])
                    grad[inds, k] = 10**(3) * (f_base2 -
                                               f_base[inds]).reshape(n_finite,)

            return grad
        thetaprior.lpdf_grad = lpdf_grad

    def logprobacce(theta):
        theta_f = 1 * theta
        for i_id in range(0, theta.shape[1]):
            for j_id in range(i_id, theta.shape[1]):
                theta_f = np.concatenate(
                    [theta_f, np.reshape(theta_f[:, i_id] * theta_f[:, j_id],
                                         (len(theta_f), 1))], axis=1)
        ml_probability = clf_method.predict_proba(theta_f)[:, 1]
        ml_logprobability = np.reshape(np.log(ml_probability),
                                       (len(theta), 1))
        return ml_logprobability

    def logprobacce_grad(theta):
        L0 = logprobacce(theta)
        dL = np.zeros((L0.shape[0], theta.shape[1]))
        for i_id in range(0, theta.shape[1]):
            theta_f = 1 * theta
            theta_f[:, i_id] += 10**(-6)
            L1 = logprobacce(theta_f)
            dL[:, i_id] = np.squeeze((L1-L0) * (10**6))
        return dL

    def logpostfull_wgrad(theta, return_grad=True):
        # obtain the log-prior
        logpost = thetaprior.lpdf(theta)
        inds = np.where(np.isfinite(logpost))[0]
        if emureturn_grad and return_grad:
            # obtain the gradient of the log-prior
            dlogpost = thetaprior.lpdf_grad(theta)
            if len(inds) > 0:
                # obtain the log-likelihood and the gradient of it
                loglikinds, dloglikinds = loglik_grad(fitinfo,
                                                      emu,
                                                      theta[inds, :],
                                                      y,
                                                      x)
                logpost[inds] += loglikinds
                dlogpost[inds] += dloglikinds
                if clf_method is not None:
                    logpost[inds] += logprobacce(theta[inds, :])
                    dlogpost[inds] += logprobacce_grad(theta[inds, :])
            return logpost, dlogpost
        else:
            if len(inds) > 0:
                # obtain the log-likelihood
                logpost[inds] += loglik(fitinfo,
                                        emu,
                                        theta[inds, :],
                                        y,
                                        x)
                if clf_method is not None:
                    logpost[inds] += logprobacce(theta[inds, :])

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
            theta0 = np.vstack((thetaprior.rnd(n-n0), theta0))
        else:
            theta0 = theta0[np.random.randint(theta0.shape[0], size=n), :]

        return theta0

    # obtain theta draws from posterior distribution
    sampler_obj = sampler(logpost_func=logpostfull_wgrad,
                          draw_func=draw_func,
                          **bayeswoodbury_args)

    theta = sampler_obj.sampler_info['theta']

    # obtain log-posterior of theta values
    ladj = logpostfull_wgrad(theta, return_grad=False)
    mladj = np.max(ladj)
    fitinfo['lpdfapproxnorm'] = np.log(np.mean(np.exp(ladj - mladj))) + mladj
    fitinfo['lpdfapproxnorm_un'] = ladj
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


def thetalpdf(fitinfo, theta, args=None):
    '''
    Returns log of the posterior of the given theta.

    Not required.
    '''

    emu = fitinfo['emu']
    y = fitinfo['y']
    x = fitinfo['x']
    thetaprior = fitinfo['thetaprior']
    logpost = thetaprior.lpdf(theta)
    if logpost.ndim > 0.5 and logpost.shape[0] > 1.5:
        inds = np.where(np.isfinite(logpost))[0]
        logpost[inds] += loglik(fitinfo, emu, theta[inds], y, x, args)
    elif np.isfinite(logpost):
        logpost += loglik(fitinfo, emu, theta, y, x, args)
    return (logpost-fitinfo['lpdfapproxnorm'])


def loglik(fitinfo, emu, theta, y, x, args=None):
    r"""
    This is a optional docstring for an internal function.
    """

    if 'yvar' in fitinfo.keys():
        obsvar = 1*np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')

    emupredict = emu.predict(x, theta)
    emumean = emupredict.mean()
    emuvar = emupredict.var()
    emucovxhalf = emupredict.covxhalf()
    loglik = np.zeros((emumean.shape[1], 1))

    if np.any(np.abs(emuvar/(10 ** (-4) +
                             (1 + 10**(-4))*np.sum(np.square(emucovxhalf),
                                                   2))) > 1):
        emuoldpredict = emu.predict(x)
        emuoldvar = emuoldpredict.var()
        emuoldcxh = emuoldpredict.covxhalf()
        obsvar += np.mean(np.abs(emuoldvar -
                                 np.sum(np.square(emuoldcxh), 2)), 1)

    # compute loglikelihood for each theta value in theta
    for k in range(0, emumean.shape[1]):
        m0 = emumean[:, k]
        S0 = np.squeeze(emucovxhalf[:, k, :])
        stndresid = (np.squeeze(y) - m0) / np.sqrt(obsvar)
        term1 = np.sum(stndresid ** 2)
        J = (S0.T / np.sqrt(obsvar)).T
        if J.ndim < 1.5:
            J = J[:, None]
            stndresid = stndresid[:, None]
        J2 = J.T @ stndresid
        W, V = np.linalg.eigh(np.eye(J.shape[1]) + J.T @ J)
        if W.shape[0] > 1:
            J3 = V @ np.diag(1/W) @ V.T @ J2
        else:
            J3 = ((V**2)/W) * J2
        term2 = np.sum(J3 * J2)
        residsq = term1 - term2
        loglik[k, 0] = -0.5 * residsq - 0.5 * np.sum(np.log(W))

    return loglik


def loglik_grad(fitinfo, emu, theta, y, x):
    r"""
    This is a optional docstring for an internal function.
    """

    if 'yvar' in fitinfo.keys():
        obsvar = 1*np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')

    emupredict = emu.predict(x, theta, args={'return_grad': True})
    emumean = emupredict.mean()
    emuvar = emupredict.var()
    emucovxhalf = emupredict.covxhalf()
    emumean_grad = emupredict.mean_gradtheta()
    emucovxhalf_grad = emupredict.covxhalf_gradtheta()

    loglik = np.zeros((emumean.shape[1], 1))
    dloglik = np.zeros((emumean.shape[1], emu._info['theta'].shape[1]))

    dterm1 = np.zeros(emu._info['theta'].shape[1])
    dterm2 = np.zeros(emu._info['theta'].shape[1])
    dterm3 = np.zeros(emu._info['theta'].shape[1])

    # adj for any unmodled variance:
    if np.any(np.abs(emuvar/(10 ** (-4) +
                             (1 + 10**(-4))*np.sum(np.square(emucovxhalf),
                                                   2))) > 1):
        emuoldpredict = emu.predict(x)
        emuoldvar = emuoldpredict.var()
        emuoldcxh = emuoldpredict.covxhalf()
        obsvar += np.mean(np.abs(emuoldvar -
                                 np.sum(np.square(emuoldcxh), 2)), 1)

    for k in range(0, emumean.shape[1]):
        m0 = emumean[:, k]
        dm0 = np.squeeze(emumean_grad[:, k, :])
        S0 = np.squeeze(emucovxhalf[:, k, :])
        stndresid = (np.squeeze(y) - m0) / np.sqrt(obsvar)
        term1 = np.sum(stndresid ** 2)
        stndresid_grad = - (dm0.T / np.sqrt(obsvar)).T
        dterm1 = 2 * np.sum(stndresid * stndresid_grad.T, 1)
        J = (S0.T / np.sqrt(obsvar)).T

        if J.ndim < 1.5:
            J = J[:, None]
        J2 = J.T @ stndresid
        W, V = np.linalg.eigh(np.eye(J.shape[1]) + J.T @ J)
        J3 = V @ np.diag(1/W) @ V.T @ J2
        term2 = np.sum(J3 * J2)

        for i in range(0, stndresid_grad.shape[1]):
            dJ = (np.squeeze(emucovxhalf_grad[:, k, :, i]).T *
                  (1/np.sqrt(obsvar)))
            dJ2 = J.T @ stndresid_grad[:, i] + dJ @ stndresid
            exmat = dJ @ J
            exmat = (exmat + exmat.T)
            dJ3 = V @ np.diag(1/W) @ V.T @ (dJ2 - exmat @ J3)
            dterm2[i] = np.sum(J2 * dJ3) + np.sum(dJ2 * J3)

        if W.shape[0] > 1:
            V2 = 1/obsvar * (((V * (1/W)) @ V.T) @ S0.T)
        else:
            V2 = (1/obsvar * ((V**2/W) * S0))
        for i in range(0, stndresid_grad.shape[1]):
            dterm3[i] = 2 * np.sum(V2 *
                                   np.squeeze(emucovxhalf_grad[:, k, :, i]).T)
        term3 = np.sum(np.log(W))
        residsq = term1 - term2
        loglik[k, 0] = - 0.5 * term3-0.5 * residsq
        dloglik[k, :] = -0.5 * dterm3 - 0.5 * (dterm1 - dterm2)

    return loglik, dloglik
