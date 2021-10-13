"""
PCGPwMatComp method - an extension of PCGP method (Higdon et al. 2008) to handle missingness in simulation data.
Matrix completion methods are used to complete the data, using matrix-completion package (Duan 2020). Then,
:py:mod:surmise.emulationmethods.PCGP is used with the completed data.
"""
import numpy as np
import surmise.emulationmethods.PCGPwM as semPCGPwM
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:  # ModuleNotFoundError introduced in Python 3.6
    print(e)
    raise ImportError('This emulation method requires installation of packages \'sklearn\' and '
                      'requires enabling of iterative imputer option.')
try:
    from matrix_completion import svt_solve, pmf_solve, biased_mf_solve
except ImportError as e:  # ModuleNotFoundError introduced in Python 3.6
    print(e)
    raise ImportError('This emulation method requires installation of packages \'matrix_completion\' '
                      'and \'cvxpy\'.')


methodoptionstr = ('\nTry one of the following: '
                   '\n\'svt\' (singular value thresholding), '
                   '\n\'pmf\' (probabilistic matrix factorization), '
                   '\n\'bmf\' (biased alterating least squares matrix factorization).')


suggeststr = '\nOtherwise, try emulation method \'PCGPwM\' to handle missing values.'


def fit(fitinfo, x, theta, f, epsilonPC=0.001, lognugmean=-10,
        lognugLB=-20, varconstant=1, dampalpha=0, bigM=1000, compmethod='KNN',
        verbose=0, **kwargs):
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

    # Check for missing or failed values
    if not np.all(np.isfinite(f)):
        fitinfo['mof'] = np.logical_not(np.isfinite(f))
        __completef(fitinfo, compmethod=compmethod)
        print('Completing f with method IterativeImputer:{:s}.'.format(compmethod))
    else:
        fitinfo['mof'] = None
        print('No missing values identified... Proceeding with PCGP method.')

    fitinfo['mof'] = None
    fitinfo['mofrows'] = None

    fitinfo['epsilonPC'] = epsilonPC
    hyp1 = lognugmean
    hyp2 = lognugLB
    hypvarconst = np.log(varconstant) if varconstant is not None else None

    fitinfo['dampalpha'] = dampalpha
    fitinfo['bigM'] = bigM

    # standardize function evaluations f
    __standardizef(fitinfo)

    # apply PCA to reduce the dimension of f
    __PCs(fitinfo)
    numpcs = fitinfo['pc'].shape[1]

    # fit a GP for each PC
    emulist = __fitGPs(fitinfo, theta, numpcs, hyp1, hyp2, hypvarconst)

    fitinfo['emulist'] = emulist
    return


def __completef(fitinfo, compmethod=None):
    """Completes missing values in f using matrix-completion library (Duan, 2020)."""
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


def __completef_matrix_completion(fitinfo, method=None):
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


# Functions below call surmise.emulationmethods.PCGPwM directly.
def predict(predinfo, fitinfo, x, theta, **kwargs):
    return semPCGPwM.predict(predinfo, fitinfo, x, theta, **kwargs)


def predictlpdf(predinfo, f, return_grad=False, addvar=0):
    return semPCGPwM.predictlpdf(predinfo, f, return_grad, addvar)


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
