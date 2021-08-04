"""
PCGPwMatComp method - an extension of PCGP method (Higdon et al. 2008) to handle missingness in simulation data.
Matrix completion methods are used to complete the data, using matrix-completion package (Duan 2020). Then,
:py:mod:surmise.emulationmethods.PCGP is used with the completed data.
"""
import numpy as np
import surmise.emulationmethods.PCGP as semPCGP
try:
    from matrix_completion import svt_solve, pmf_solve, biased_mf_solve
except ImportError:  # ModuleNotFoundError introduced in Python 3.6
    raise ImportError('This emulation method requires installation of packages \'matrix_completion\' and \'cvxpy\'.')


methodoptionstr = ('\nTry one of the following: '
                   '\n\'svt\' (singular value thresholding), '
                   '\n\'pmf\' (probabilistic matrix factorization), '
                   '\n\'bmf\' (biased alterating least squares matrix factorization).')


suggeststr = '\nOtherwise, try emulation method \'PCGPwM\' to handle missing values.'


def fit(fitinfo, x, theta, f, epsilon=0.1, completionmethod='svt', **kwargs):
    """
    The purpose of fit is to take information and plug all of our fit
    information into fitinfo, which is a python dictionary.

    Parameters
    ----------
    fitinfo : dict
        A dictionary including the emulation fitting information once
        complete.
        The dictionary is passed by reference, so it returns None.
    x : numpy.ndarray
        An array of inputs. Each row should correspond to a row in f.
    theta : numpy.ndarray
        An array of parameters. Each row should correspond to a column in f.
    f : numpy.ndarray
        An array of responses. Each column in f should correspond to a row in
        theta. Each row in f should correspond to a row in x.
    completionmethod : str
        A string variable containing the name of matrix completion method. Options
        are:
        - \'svt\' (singular value thresholding),
        - \'pmf\' (probabilistic matrix factorization),
        - \'bmf\' (biased alterating least squares matrix factorization).

    Returns
    -------
    None.

    """
    f = f.T
    fitinfo['theta'] = theta
    fitinfo['f'] = f
    fitinfo['x'] = x
    fitinfo['epsilon'] = epsilon

    # Check for missing or failed values
    if not np.all(np.isfinite(f)):
        fitinfo['mof'] = np.logical_not(np.isfinite(f))
        print('Completing f with method \'{:s}\''.format(completionmethod))
        __completef(fitinfo, completionmethod)
    else:
        fitinfo['mof'] = None
        print('No missing values identified... Proceeding with PCGP method.')

    # standardize function evaluations f
    __standardizef(fitinfo)

    # apply PCA to reduce the dimension of f
    __PCs(fitinfo)
    numpcs = fitinfo['pc'].shape[1]

    # create a dictionary to save the emu info for each PC
    emulist = [dict() for x in range(0, numpcs)]

    # fit a GP for each PC
    for pcanum in range(0, numpcs):
        emulist[pcanum] = emulation_fit(theta, fitinfo['pc'][:, pcanum])

    fitinfo['emulist'] = emulist
    return


def __completef(fitinfo, method=None):
    """Completes missing values in f using matrix-completion library (Duan, 2020)."""
    mof = fitinfo['mof']
    f = fitinfo['f']

    if method is None or method == 'svt':
        # Singular value thresholding
        fhat = svt_solve(f, ~mof)
    elif method == 'pmf':
        # Probablistic matrix factorization
        fhat = pmf_solve(f, ~mof, k=10, mu=1e-2)
    elif method == 'bmf':
        # Biased alternating least squares
        fhat = biased_mf_solve(f, ~mof, k=10, mu=1e-2)
    else:
        raise ValueError('Unsupported completion method. {:s}'.format(methodoptionstr))

    if not np.isfinite(fhat).all():
        raise ValueError('Completion method {:s} failed. {:s} {:s}'.format(method, methodoptionstr, suggeststr))

    fitinfo['f'] = fhat
    fitinfo['completionmethod'] = method
    return


# Functions below call surmise.emulationmethods.PCGP directly.
def predict(predinfo, fitinfo, x, theta, computecov=True, **kwargs):
    '''
    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where you should place all of your prediction
        information once complete.

        - predinfo['mean'] : mean prediction.

        - predinfo['cov'] : variance of the prediction.

    x : numpy.ndarray
        An array of inputs. Each row should correspond to a row in f.

    theta : numpy.ndarray
        An array of parameters. Each row should correspond to a column in f.

    f : numpy.ndarray
        An array of responses. Each column in f should correspond to a row in
        x. Each row in f should correspond to a row in x.

    args : dict, optional
        A dictionary containing options. The default is None.

    Returns
    -------
    Prediction mean and variance at theta and x given the dictionary fitinfo.
    '''

    return semPCGP.predict(predinfo, fitinfo, x, theta, computecov)


def predictmean(predinfo, **kwargs):
    return predinfo['mean']


def predictvar(predinfo, **kwargs):
    return predinfo['var']


def emulation_covmat(theta1, theta2, gammav, returndir=False):
    '''
    Returns covariance matrix R (Matern) such that

    .. math::
        R = (1 + S)*\\exp(-S)

    where

    .. math::
        S = \\frac{|\\theta_1 - \\theta_2|}{\\gamma}.

    The matrix inverse is obtained via eigendecomposition

    .. math::
        R^{-1} = V (W^{-1}) V^T

    where V is eigenvector and W are eigenvalues.

    Parameters
    ----------
    theta1 : numpy.ndarray
        An n1-by-d array of parameters.
    theta2 : numpy.ndarray
       An n2-by-d array of parameters.
    gammav : numpy.ndarray
        An array of length d covariance hyperparameters
    returndir : Bool, optional
        Boolean. If True, returns dR. The default is False.

    Returns
    -------
    numpy.ndarray
        A n1-by-n2 array of covariance between theta1 and theta2 given
        parameter gammav.

    '''

    return semPCGP.emulation_covmat(theta1, theta2, gammav, returndir)


def emulation_negloglik(hyperparameters, fitinfo):
    '''
    Returns the negative log-likelihood of a univariate GP model for given
    hyperparameters. Hyperparameters minimize the following

    .. math::

        \\frac{n}{2}\\log{\\hat{\\sigma}^2} + \\frac{1}{2} \\log |R|.

    Parameters
    ----------
    hyperparameters : numpy.ndarray
        An array of hyperparameters.
    fitinfo : dict
        A dictionary including the emulation fitting information.

    Returns
    -------
    negloglik : float
           Negative log-likelihood of a univariate GP model.

    '''

    return semPCGP.emulation_negloglik(hyperparameters, fitinfo)


def emulation_negloglikgrad(hyperparameters, fitinfo):
    '''
    Parameters
    ----------
    hyperparameters : numpy.ndarray
        An array of hyperparameters.
    fitinfo : dict
        A dictionary including the emulation fitting information.

    Returns
    -------
    dnegloglik : float
        Gradient of the log-likelihood of a univariate GP model.

    '''

    return semPCGP.emulation_negloglikgrad(hyperparameters, fitinfo)


def emulation_fit(theta, pcaval):
    '''
    Fits a univariate GP. First, obtains the hyperparameter values via
    'L-BFGS-B'. Then, finds the MLEs of mu and sigma^2 for the best
    hyperparameters such that

    .. math::
        \\hat{\\mu} = \\frac{1^T R^{-1} f}{1^T R^{-1} 1}

    .. math::
        \\hat{\\sigma^2} =
        \\frac{(f-\\hat{\\mu})^T R^{-1} (f-\\hat{\\mu})}{n}

    Parameters
    ----------
    theta : numpy.ndarray
         An n-by-d array of parameters.
    pcaval : numpy.ndarray
        An array of length n.

    Returns
    -------
    subinfo : dict
        Dictionary of the fitted emulator model.
    '''

    return semPCGP.emulation_fit(theta, pcaval)


def __standardizef(fitinfo, offset=None, scale=None):
    """Standardizes f by creating offset, scale and fs."""
    return semPCGP.__standardizef(fitinfo, offset, scale)


def __PCs(fitinfo):
    """Apply PCA to reduce the dimension of f."""
    return semPCGP.__PCs(fitinfo)
