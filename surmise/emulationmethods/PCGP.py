"""PCGP (Higdon et al., 2008)"""
import numpy as np
import scipy.optimize as spo


def fit(fitinfo, x, theta, f, args=None):
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
        x. Each row in f should correspond to a row in x.
    args : dict, optional
        A dictionary containing options. The default is None.

    Returns
    -------
    None.

    '''

    f = f.T
    fitinfo['theta'] = theta
    fitinfo['x'] = x
    fitinfo['f'] = f
    fitinfo['epsilon'] = 1

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


def predict(predinfo, fitinfo, x, theta, args=None):
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

    infos = fitinfo['emulist']
    predvecs = np.zeros((theta.shape[0], len(infos)))
    predvars = np.zeros((theta.shape[0], len(infos)))

    # xind = range(0, fitinfo['x'].shape[0])
    try:
        if x is None or np.all(np.equal(x, fitinfo['x'])) or \
                np.allclose(x, fitinfo['x']):
            xind = np.arange(0, x.shape[0])
            xnewind = np.arange(0, x.shape[0])
        else:
            raise
    except Exception:
        matchingmatrix = np.ones((x.shape[0], fitinfo['x'].shape[0]))
        for k in range(0, x[0].shape[0]):
            try:
                matchingmatrix *= np.isclose(x[:, k][:, None],
                                             fitinfo['x'][:, k])
            except Exception:
                matchingmatrix *= np.equal(x[:, k][:, None],
                                           fitinfo['x'][:, k])
        xind = np.argwhere(matchingmatrix > 0.5)[:, 1]
        xnewind = np.argwhere(matchingmatrix > 0.5)[:, 0]

    # For each PC, obtain the mean and variance
    for k in range(0, len(infos)):
        r = emulation_covmat(theta, fitinfo['theta'], infos[k]['hypcov'])
        predvecs[:, k] = infos[k]['muhat'] + r @ infos[k]['pw']
        Rinv = infos[k]['Rinv']
        predvars[:, k] = infos[k]['sigma2hat'] * \
            (1 + np.exp(infos[k]['hypnug']) - np.sum(r.T * (Rinv @ r.T), 0))

    pctscale = (fitinfo['pct'].T * fitinfo['scale']).T

    # transfer back the PCs into the original space
    predinfo['mean'] = np.full((x.shape[0], theta.shape[0]), np.nan)
    predinfo['var'] = np.full((x.shape[0], theta.shape[0]), np.nan)
    predinfo['mean'][xnewind, :] = (predvecs @ pctscale[xind, :].T +
                                    fitinfo['offset'][xind]).T
    predinfo['var'][xnewind, :] = (fitinfo['extravar'][xind] +
                                   (predvars @ pctscale[xind, :].T ** 2)).T

    CH = (np.sqrt(predvars)[:, :, None] * (pctscale[xind, :].T)[None, :, :])
    predinfo['covxhalf'] = np.full((theta.shape[0],
                                    CH.shape[1],
                                    x.shape[0]), np.nan)
    predinfo['covxhalf'][:, :, xnewind] = CH
    predinfo['covxhalf'] = predinfo['covxhalf'].transpose((2, 0, 1))

    return


def predictmean(predinfo, args=None):
    return predinfo['mean']


def predictvar(predinfo, args=None):
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
    d = gammav.shape[0]
    theta1 = theta1.reshape(1, d) if theta1.ndim < 1.5 else theta1
    theta2 = theta2.reshape(1, d) if theta2.ndim < 1.5 else theta2
    n1 = theta1.shape[0]
    n2 = theta2.shape[0]
    V = np.zeros([n1, n2])
    R = np.ones([n1, n2])

    # Matern covariance structure, that is, R = (1 + S)*exp(-S)
    if returndir:
        dR = np.zeros([n1, n2, d])
    for k in range(0, d):
        S = np.abs(np.subtract.outer(theta1[:, k], theta2[:, k]) /
                   np.exp(gammav[k]))
        R *= (1 + S)
        V -= S
        if returndir:
            dR[:, :, k] = (S * S) / (1 + S)
    R *= np.exp(V)
    if returndir:
        for k in range(0, d):
            dR[:, :, k] = R * dR[:, :, k]
        return R, dR
    else:
        return R


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
    # obtain the hyperparameter values
    covhyp = hyperparameters[0:fitinfo['p']]
    nughyp = hyperparameters[fitinfo['p']]

    # get the fitinfo values
    theta = fitinfo['theta']
    n = fitinfo['n']
    f = fitinfo['f']

    # obtain the covariance matrix R
    R = emulation_covmat(theta, theta, covhyp)
    R = R + np.exp(nughyp)*np.diag(np.ones(n))

    # eigendecomposition of R
    W, V = np.linalg.eigh(R)

    # MLEs for mu and sigma^2
    fspin = V.T @ f
    onespin = V.T @ np.ones(f.shape)
    muhat = np.sum(V @ (1/W * fspin)) / np.sum(V @ (1/W * onespin))
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)

    # Negative log-likelihood of a univariate GP model
    negloglik = 1/2 * np.sum(np.log(W)) + n/2 * np.log(sigma2hat)
    negloglik += 1/2 * np.sum((hyperparameters - fitinfo['hypregmean'])**2 /
                              (fitinfo['hypregstd'] ** 2))
    return negloglik


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
    # obtain the hyperparameter values
    covhyp = hyperparameters[0:fitinfo['p']]
    nughyp = hyperparameters[fitinfo['p']]

    # get the fitinfo values
    theta = fitinfo['theta']
    n = fitinfo['n']
    p = fitinfo['p']
    f = fitinfo['f']

    # obtain the covariance matrix
    R, dR = emulation_covmat(theta, theta, covhyp, True)
    R = R + np.exp(nughyp)*np.diag(np.ones(n))
    dRappend = np.exp(nughyp)*np.diag(np.ones(n)).reshape(R.shape[0],
                                                          R.shape[1], 1)
    dR = np.append(dR, dRappend, axis=2)

    # MLEs for mu and sigma^2
    W, V = np.linalg.eigh(R)
    fspin = V.T @ f
    onespin = V.T @ np.ones(f.shape)
    mudenom = np.sum(V @ (1/W * onespin))
    munum = np.sum(V @ (1/W * fspin))
    muhat = munum / mudenom
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)

    # gradients
    dmuhat = np.zeros(p + 1)
    dsigma2hat = np.zeros(p + 1)
    dfcentercalc = (fcenter / W) @ V.T
    dfspincalc = (fspin / W) @ V.T
    donespincalc = (onespin / W) @ V.T
    Rinv = V @ np.diag(1/W) @ V.T
    dlogdet = np.zeros(p + 1)

    for k in range(0, dR.shape[2]):
        dRnorm = np.squeeze(dR[:, :, k])
        dmuhat[k] = -np.sum(donespincalc @ dRnorm @ dfspincalc) \
            / mudenom + \
            muhat * (np.sum(donespincalc @ dRnorm @ donespincalc)/mudenom)
        dsigma2hat[k] = -(1/n) * (dfcentercalc.T @ dRnorm @ dfcentercalc) + \
            2*dmuhat[k] * np.mean((fcenter * onespin) / W)
        dlogdet[k] = np.sum(Rinv * dRnorm)

    # Gradient of the log-likelihood of a single dimensional GP model.
    dnegloglik = 1/2 * dlogdet + n/2 * 1/sigma2hat * dsigma2hat
    dnegloglik += ((hyperparameters - fitinfo['hypregmean']) /
                   (fitinfo['hypregstd'] ** 2))
    return dnegloglik


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
    subinfo = {}

    covhyp0 = np.log(np.std(theta, 0)*3) + 1
    covhypLB = covhyp0 - 2
    covhypUB = covhyp0 + 3

    nughyp0 = -6
    nughypLB = -15
    nughypUB = 1

    # Get a random sample of thetas to find the optimized hyperparameters
    n_train = np.min((20*theta.shape[1], theta.shape[0]))
    idx = np.random.choice(theta.shape[0], n_train, replace=False)

    # Start constructing the returning dictionary
    if theta.ndim == 1:
        subinfo['theta'] = theta[idx]
    else:
        subinfo['theta'] = theta[idx, :]
    subinfo['f'] = pcaval[idx]
    subinfo['n'] = subinfo['f'].shape[0]
    subinfo['p'] = covhyp0.shape[0]

    subinfo['hypregmean'] = np.append(covhyp0, nughyp0)
    subinfo['hypregstd'] = np.append((covhypUB - covhypLB)/3, 4)

    # Find the hyperparameters minimizing the negative loglikelihood
    bounds = spo.Bounds(np.append(covhypLB, nughypLB),
                        np.append(covhypUB, nughypUB))
    opval = spo.minimize(emulation_negloglik,
                         np.append(covhyp0, nughyp0),
                         bounds=bounds,
                         args=(subinfo),
                         method='L-BFGS-B',
                         options={'disp': False},
                         jac=emulation_negloglikgrad)

    # Obtain the optimized hyperparameter values
    hypcov = opval.x[:subinfo['p']]
    hypnug = opval.x[subinfo['p']]

    # Obtain the covariance matrix
    R = emulation_covmat(theta, theta, hypcov)
    R = R + np.exp(hypnug)*np.diag(np.ones(R.shape[0]))

    # Obtain the eigenvalue decomposition of the covariance matrix
    W, V = np.linalg.eigh(R)

    # MLEs for mu and sigma^2
    fspin = V.T @ pcaval
    onespin = V.T @ np.ones(pcaval.shape)
    muhat = np.sum(V @ (1/W * fspin)) / np.sum(V @ (1/W * onespin))
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)

    # Obtain the inverse of the covariance matrix
    Rinv = V @ np.diag(1/W) @ V.T

    # Construct the dictionary with the fitted emulator
    subinfo['hypcov'] = hypcov
    subinfo['hypnug'] = hypnug
    subinfo['R'] = R
    subinfo['Rinv'] = Rinv
    subinfo['pw'] = Rinv @ (pcaval - muhat)
    subinfo['muhat'] = muhat
    subinfo['sigma2hat'] = sigma2hat
    subinfo['theta'] = theta

    return subinfo


def __standardizef(fitinfo, offset=None, scale=None):
    "Standardizes f by creating offset, scale and fs."
    # Extracting from input dictionary
    f = fitinfo['f']

    if (offset is not None) and (scale is not None):
        if offset.shape[0] == f.shape[1] and scale.shape[0] == f.shape[1]:
            if np.any(np.nanmean(np.abs(f-offset)/scale, 1) > 4):
                offset = None
                scale = None
        else:
            offset = None
            scale = None
    if offset is None or scale is None:
        offset = np.zeros(f.shape[1])
        scale = np.zeros(f.shape[1])
        for k in range(0, f.shape[1]):
            offset[k] = np.nanmean(f[:, k])
            scale[k] = np.nanstd(f[:, k])
            if scale[k] == 0:
                scale[k] = 0.0001

    # Initializing values
    fs = np.zeros(f.shape)
    fs = (f - offset) / scale

    # Assigning new values to the dictionary
    fitinfo['offset'] = offset
    fitinfo['scale'] = scale
    fitinfo['fs'] = fs
    return


def __PCs(fitinfo):
    "Apply PCA to reduce the dimension of f"
    # Extracting from input dictionary
    f = fitinfo['f']
    fs = fitinfo['fs']
    epsilon = fitinfo['epsilon']
    pct = None
    pcw = None

    U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
    Sp = S ** 2 - epsilon
    pct = U[:, Sp > 0]
    pcw = np.sqrt(Sp[Sp > 0])
    pcstdvar = np.zeros((f.shape[0], pct.shape[1]))

    fitinfo['pcw'] = pcw
    fitinfo['pcto'] = 1*pct
    fitinfo['pct'] = pct * pcw / np.sqrt(pct.shape[0])
    fitinfo['pcti'] = pct * (np.sqrt(pct.shape[0]) / pcw)
    fitinfo['pc'] = fs @ fitinfo['pcti']
    fitinfo['extravar'] = np.mean((fs - fitinfo['pc'] @
                                   fitinfo['pct'].T) ** 2, 0) *\
        (fitinfo['scale'] ** 2)
    fitinfo['pcstdvar'] = 10*pcstdvar
    return
