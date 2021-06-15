"""MODULE DESCRIPTION"""
import numpy as np
import scipy.optimize as spo
import surmise.emulationmethods.PCGP as semPCGP
import matrix_completion


def fit(fitinfo, x, theta, f, epsilon=0.1, **kwargs):
    '''
    The purpose of fit is to take information and plug all of our fit
    information into fitinfo, which is a python dictionary.

    .. note::
       This is an application of the method proposed by Higdon et al., 2008.
       The idea is to use PCA to project the original simulator outputs
       onto a lower-dimensional space spanned by an orthogonal basis. The main
       steps are

        - 1. Standardize f
        - 2. Compute the SVD of f, and get the PCs
        - 3. Project the original centred data into the orthonormal space to
          obtain the matrix of coefficients (say we use r PCs)
        - 4. Then, build r separate and independent GPs from the input space

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
    args : dict, optional
        A dictionary containing options. The default is None.

    Returns
    -------
    None.

    '''

    f = f.T
    # Check for missing or failed values
    if not np.all(np.isfinite(f)):
        fitinfo['mof'] = np.logical_not(np.isfinite(f))
    else:
        fitinfo['mof'] = None

    fitinfo['theta'] = theta
    fitinfo['f'] = f
    fitinfo['x'] = x

    return


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


    return


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

def __completef(fitinfo, method=None):
    """"""
    if 'mof' not in fitinfo.keys():
        raise KeyError('No missing ')

def __standardizef(fitinfo, offset=None, scale=None):
    "Standardizes f by creating offset, scale and fs."
    # Extracting from input dictionary


def __PCs(fitinfo):
    "Apply PCA to reduce the dimension of f"