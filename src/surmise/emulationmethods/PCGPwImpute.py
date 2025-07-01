"""
PCGPwImpute method - an extension of PCGP method (Higdon et al. 2008) to handle missingness in simulation data.
Missing values are first imputed using methods included in IterativeImputer in scikit-learn.
Then, :py:mod:surmise.emulationmethods.PCGPwM is used with the completed data.
"""
import numpy as np
import surmise.emulationmethods.PCGPwM as semPCGPwM

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

methodoptionstr = ('\nTry one of the following: '
                   '\n\'KNN\' (k-nearest neighbor method), '
                   '\n\'BayesianRidge\' (Bayesian ridge regression), '
                   '\n\'RandomForest\' (random forest method).')

suggeststr = '\nOtherwise, try emulation method \'PCGPwM\' to handle missing values.'


def fit(fitinfo, x, theta, f, epsilonPC=0.001, lognugmean=-10,
        lognugLB=-20, completionmethod='KNN', verbose=0, **kwargs):
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
    epsilonPC : scalar
        A parameter to control the number of PCs used.  The suggested range for
        epsilonPC is (0.001, 0.1).  The larger epsilonPC is, the fewer PCs will be
        used.  Note that epsilonPC here is *not* the unexplained variance in
        typical principal component analysis.
    completionmethod : str
        A string variable containing the name of matrix completion method. Options
        are:
        - \'KNN\' (k-nearest neighbor method),
        - \'BayesianRidge\' (Bayesian ridge regression),
        - \'RandomForest\' (random forest method).

    Returns
    -------
    None.

    """
    f = f.T
    fitinfo['theta'] = theta
    fitinfo['f'] = f
    fitinfo['x'] = x

    # Check for missing or failed values
    if not np.all(np.isfinite(f)):
        fitinfo['mof'] = np.logical_not(np.isfinite(f))
        __completef(fitinfo, compmethod=completionmethod)
        if verbose > 0:
            print('Completing f with method IterativeImputer:{:s}.'.format(completionmethod))
    else:
        fitinfo['mof'] = None
        fitinfo['completionmethod'] = ''
        if verbose > 0:
            print('No missing values identified... Proceeding with PCGP method.')

    fitinfo['mof'] = None
    fitinfo['mofrows'] = None

    fitinfo['epsilonPC'] = epsilonPC

    hyp1 = lognugmean
    hyp2 = lognugLB

    # parameters that are features of PCGPwM, but irrelevant with PCGP with imputation
    varconstant = 1
    dampalpha = 0
    eta = 1000
    hypvarconst = np.log(varconstant) if varconstant is not None else None
    fitinfo['dampalpha'] = dampalpha
    fitinfo['eta'] = eta

    # standardize function evaluations f
    __standardizef(fitinfo)

    # apply PCA to reduce the dimension of f
    __PCs(fitinfo)
    numpcs = fitinfo['pc'].shape[1]

    # fit a GP for each PC
    emulist = __fitGPs(fitinfo, theta, numpcs, hyp1, hyp2, hypvarconst)

    fitinfo['emulist'] = emulist

    __generate_param_str(fitinfo)
    return


def __completef(fitinfo, compmethod=None):
    """Completes missing values in f using IterativeImpute from scikit-learn."""
    f = fitinfo['f']
    if compmethod == 'KNN':
        estimator = KNeighborsRegressor()
    elif compmethod == 'BayesianRidge':
        estimator = BayesianRidge()
    elif compmethod == 'RandomForest':
        estimator = RandomForestRegressor()
    else:
        raise ValueError('Specify completion method.')

    transformer = IterativeImputer(estimator=estimator)
    fhat = transformer.fit_transform(f)

    if not np.isfinite(fhat).all():
        raise ValueError('Completion method (sklearn.impute.IterativeImputer) failed.')

    fitinfo['f'] = fhat
    fitinfo['completionmethod'] = 'IterativeImputer:{:s}'.format(compmethod)
    return


# Functions below call surmise.emulationmethods.PCGPwM directly.
def predict(predinfo, fitinfo, x, theta, **kwargs):
    return semPCGPwM.predict(predinfo, fitinfo, x, theta, **kwargs)


def predictlpdf(predinfo, f, addvar=0, **kwargs):
    return semPCGPwM.predictlpdf(predinfo, f, addvar, **kwargs)


def __fitGPs(fitinfo, theta, numpcs, hyp1, hyp2, varconstant):
    return semPCGPwM.__fitGPs(fitinfo, theta, numpcs, hyp1, hyp2, varconstant)


def __fitGP1d(theta, g, hyp1, hyp2, hypvarconst, gvar=None, dampalpha=None, hypstarts=None, hypinds=None):
    return semPCGPwM.__fitGP1d(theta, g, hyp1, hyp2, hypvarconst, gvar, dampalpha, hypstarts, hypinds)


def __negloglik(hyp, info):
    return semPCGPwM.__negloglik(hyp, info)


def __negloglikgrad(hyp, info):
    return semPCGPwM.__negloglikgrad(hyp, info)


def __standardizef(fitinfo, offset=None, scale=None):
    """Standardizes f by creating offset, scale and fs."""
    return semPCGPwM.__standardizef(fitinfo, offset, scale)


def __PCs(fitinfo):
    """Apply PCA to reduce the dimension of f."""
    return semPCGPwM.__PCs(fitinfo)


def __generate_param_str(fitinfo):
    semPCGPwM.__generate_param_str(fitinfo)
    fitinfo['param_desc'] = '\tcompletion method: {:s}\n{:s}'.format(
        fitinfo['completionmethod'], fitinfo['param_desc'])
    return
