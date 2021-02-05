"""Emulator PCGP"""

import numpy as np
import scipy.optimize as spo


def fit(fitinfo, x, theta, f, args=None):
    '''
    The purpose of fit is to take information and plug all of our fit
    information into fitinfo, which is a python dictionary.

    .. note::
       This is an application of the method proposed by Higdon et al..
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
    fitinfo['offset'] = np.zeros(f.shape[1])
    fitinfo['scale'] = np.ones(f.shape[1])
    fitinfo['theta'] = theta
    fitinfo['x'] = x

    # Standardize the function evaluations f
    for k in range(0, f.shape[1]):
        fitinfo['offset'][k] = np.mean(f[:, k])
        fitinfo['scale'][k] = np.std(f[:, k])
        if fitinfo['scale'][k] == 0:
            fitinfo['scale'][k] = 0.0001
        # fitinfo['scale'][k] = 0.9*np.std(f[:, k]) + 0.1*np.std(f)

    fstand = (f - fitinfo['offset']) / fitinfo['scale']

    # Do PCA to reduce the dimension of the function evaluations
    Vecs, Vals, _ = np.linalg.svd((fstand / np.sqrt(fstand.shape[0])).T)
    Vals = np.append(Vals, np.zeros(Vecs.shape[1] - Vals.shape[0]))
    Valssq = (fstand.shape[0]*(Vals**2) + 0.001) / (fstand.shape[0] + 0.001)

    # Find the best size of the reduced space

    numVals = 1 + np.sum(np.cumsum(Valssq) < 0.9995*np.sum(Valssq))
    numVals = np.maximum(np.minimum(2, fstand.shape[1]), numVals)

    #
    fitinfo['Cs'] = Vecs * np.sqrt(Valssq)
    fitinfo['PCs'] = fitinfo['Cs'][:, :numVals]
    fitinfo['PCsi'] = Vecs[:, :numVals] * np.sqrt(1 / Valssq[:numVals])

    pcaval = fstand @ fitinfo['PCsi']
    fhat = pcaval @ fitinfo['PCs'].T
    fitinfo['extravar'] = np.mean((fstand - fhat) ** 2,
                                  0) * (fitinfo['scale'] ** 2)

    # create a dictionary to save the emu info for each PC
    emulist = [dict() for x in range(0, numVals)]

    print(fitinfo['method'], 'considering ', numVals, 'PCs')

    # fit an emulator for each pc
    for pcanum in range(0, numVals):
        emulist[pcanum] = emulation_fit(theta, pcaval[:, pcanum])

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

    xind = range(0, fitinfo['x'].shape[0])

    # For each PC, obtain the mean and variance
    for k in range(0, len(infos)):
        r = emulation_covmat(theta, fitinfo['theta'], infos[k]['hypcov'])
        predvecs[:, k] = infos[k]['muhat'] + r @ infos[k]['pw']
        Rinv = infos[k]['Rinv']
        predvars[:, k] = infos[k]['sigma2hat'] * \
            (1 + np.exp(infos[k]['hypnug']) - np.sum(r.T * (Rinv @ r.T), 0))

    # Transfer back the PCs into the original space
    predmean = (predvecs @ fitinfo['PCs'][xind, :].T) * \
        fitinfo['scale'][xind] + fitinfo['offset'][xind]

    predvar = fitinfo['extravar'][xind] + \
        (predvars @ (fitinfo['PCs'][xind, :] ** 2).T) * \
        (fitinfo['scale'][xind] ** 2)

    predinfo['mean'] = predmean.T
    predinfo['var'] = predvar.T

    return


def predictmean(predinfo, args=None):
    return predinfo['mean']


def predictvar(predinfo, args=None):
    return predinfo['var']


def emulation_covmat(theta1, theta2, gammav, returndir=False):
    '''
    Parameters
    ----------
    theta1 : Array
        An n1-by-d array of parameters.
    theta2 : Array
       An n2-by-d array of parameters.
    gammav : Array
        A length d array.
    returndir : Bool, optional
        Boolean. If True, returns dR. The default is False.

    Returns
    -------
    Array
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
    Parameters
    ----------
    hyperparameters : TYPE
        DESCRIPTION.
    fitinfo : TYPE
        DESCRIPTION.

    Returns
    -------
    negloglik : TYPE
           Negative log-likelihood of single demensional GP model.

    '''
    # Obtain the hyperparameter values
    covhyp = hyperparameters[0:fitinfo['p']]
    nughyp = hyperparameters[fitinfo['p']]

    # Set the fitinfo values
    theta = fitinfo['theta']
    n = fitinfo['n']
    f = fitinfo['f']

    # Obtain the covariance matrix
    R = emulation_covmat(theta, theta, covhyp)
    R = R + np.exp(nughyp)*np.diag(np.ones(n))

    #
    W, V = np.linalg.eigh(R)
    fspin = V.T @ f
    onespin = V.T @ np.ones(f.shape)
    muhat = np.sum(V @ (1/W * fspin)) / np.sum(V @ (1/W * onespin))
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)

    # Negative log-likelihood of a single dimensional GP model
    negloglik = 1/2 * np.sum(np.log(W)) + n/2 * np.log(sigma2hat)
    negloglik += 1/2 * np.sum((hyperparameters - fitinfo['hypregmean'])**2 /
                              (fitinfo['hypregstd'] ** 2))
    return negloglik


def emulation_negloglikgrad(hyperparameters, fitinfo):
    '''
    Parameters
    ----------
    hyperparameters : TYPE
        DESCRIPTION.
    fitinfo : TYPE
        DESCRIPTION.

    Returns
    -------
    dnegloglik : TYPE
        Gradient of the log-likelihood of a single dimensional GP model.

    '''
    # Obtain the hyper-parameter values
    covhyp = hyperparameters[0:fitinfo['p']]
    nughyp = hyperparameters[fitinfo['p']]

    # Set the fitinfo values
    theta = fitinfo['theta']
    n = fitinfo['n']
    p = fitinfo['p']
    f = fitinfo['f']

    # Obtain the covariance matrix
    R, dR = emulation_covmat(theta, theta, covhyp, True)
    R = R + np.exp(nughyp)*np.diag(np.ones(n))
    dRappend = np.exp(nughyp)*np.diag(np.ones(n)).reshape(R.shape[0],
                                                          R.shape[1], 1)
    dR = np.append(dR, dRappend, axis=2)

    #
    W, V = np.linalg.eigh(R)
    fspin = V.T @ f
    onespin = V.T @ np.ones(f.shape)
    mudenom = np.sum(V @ (1/W * onespin))
    munum = np.sum(V @ (1/W * fspin))
    muhat = munum / mudenom
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)

    #
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


def emulation_fit(theta, pcaval, hypstarts=None, hypinds=None):
    '''
    Parameters
    ----------
    theta : Array
         An n-by-d array of parameters.
    pcaval : Array
        An array of length n.
    hypstarts : TYPE, optional
        DESCRIPTION. The default is None.
    hypinds : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    subinfo : Dictionary
        Dictionary of the fitted emulator model.
    '''
    subinfo = {}

    covhyp0 = np.log(np.std(theta, 0)*3) + 1
    covhypLB = covhyp0 - 2
    covhypUB = covhyp0 + 3

    # nughyp0 = -6
    # nughypLB = -8
    # nughypUB = 1

    nughyp0 = -6
    nughypLB = -15
    nughypUB = 5

    # Get a random sample of thetas to find the optimized hyperparameters
    # n_train = np.min((20*theta.shape[1], theta.shape[0]))
    n_train = np.min((10, theta.shape[0]))
    idx = np.random.choice(theta.shape[0], n_train, replace=False)

    # Start constructing the returning dictionary
    if theta.ndim == 1:
        subinfo['theta'] = theta[idx]
    else:
        subinfo['theta'] = theta[idx, :]
    subinfo['f'] = pcaval[idx]
    subinfo['n'] = subinfo['f'].shape[0]
    subinfo['p'] = covhyp0.shape[0]
    # TO MATT: dont know why we set them like that. we should do it optional
    subinfo['hypregmean'] = np.append(covhyp0, nughyp0)
    subinfo['hypregstd'] = np.append((covhypUB - covhypLB)/3, 1)

    # Run an optimizer to find the hyperparameters minimizing the negative
    # likelihood
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
