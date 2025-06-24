"""indGP method constructs an independent GP for every location. """
import warnings
import numpy as np
import scipy.optimize as spo
from surmise.emulationsupport.matern_covmat import covmat as __covmat
from pprint import pformat


def fit(fitinfo, x, theta, f,
        lognugmean=-10, lognugLB=-20, **kwargs):
    '''
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
    lognugmean : scalar
        A parameter to control the log of the nugget used in fitting the GPs.
        The suggested range for lognugmean is (-12, -4).  The nugget is estimated,
        and this parameter is used to guide the estimation.
    lognugLB : scalar
        A parameter to control the lower bound of the log of the nugget. The
        suggested range for lognugLB is (-24, -12).
    kwargs : dict, optional
        A dictionary containing options. The default is None.

    Returns
    -------
    None.

    '''
    # Check for missing or failed values
    fitinfo['mof'] = np.logical_not(np.isfinite(f))
    fitinfo['mofrows'] = np.where(np.any(fitinfo['mof'] > 0.5, 1))[0]
    fitinfo['mofall'] = np.where(np.all(fitinfo['mof'], 1))[0]

    if len(fitinfo['mofall']) > 0:
        warnings.warn("""Some rows are completely missing. An emulator will still be built,
                      but will return NaN predictions at those corresponding locations.
                      If you are to proceed with calibration with surmise, remove these
                      rows.""", stacklevel=2)

    nx = x.shape[0]

    hyp1 = lognugmean
    hyp2 = lognugLB
    fitinfo['theta'] = theta
    fitinfo['f'] = f
    fitinfo['x'] = x
    fitinfo['nx'] = nx

    # Standardize the function evaluations f
    __standardizef(fitinfo)

    # Fit emulators for all locations
    emulist = __fitGPs(fitinfo, theta, nx, hyp1, hyp2)
    fitinfo['emulist'] = emulist

    __generate_param_str(fitinfo)
    return


def __standardizef(fitinfo):
    r'''Standardizes f by creating offset, scale and fs.'''
    # Extracting from input dictionary
    f = fitinfo['f']
    nx = fitinfo['nx']
    mof = fitinfo['mof']
    mofall = fitinfo['mofall']

    buildinds = np.delete(np.arange(nx), mofall)
    fitinfo['buildinds'] = buildinds

    offset = np.full(nx, np.nan)
    scale = np.full(nx, np.nan)

    if mof is not None:
        for k in buildinds:
            offset[k] = np.nanmean(f[k])
            scale[k] = np.nanstd(f[k])  # / np.sqrt(1-np.isnan(f[:, k]).mean())
    else:
        offset = np.mean(f, 1)
        scale = np.std(f, 1)
    if (scale == 0).any():
        raise ValueError('One of the rows in f is non-varying.')

    fs = ((f.T - offset) / scale).T

    fitinfo['offset'] = offset
    fitinfo['scale'] = scale
    fitinfo['fs'] = fs
    return


def __fitGPs(fitinfo, theta, nx, hyp1, hyp2):
    """Fit emulators for all locations (x)."""
    emulist = [dict() for x in range(nx)]
    for j in fitinfo['buildinds']:
        mask_j = fitinfo['mof'][j]
        emulist[j] = __fitGP1d(theta=theta[~mask_j],
                               g=fitinfo['fs'][j, ~mask_j],
                               hyp1=hyp1,
                               hyp2=hyp2)
    return emulist


def __fitGP1d(theta, g, hyp1, hyp2, sig2ofconst=1):
    """Return a fitted GP model."""
    subinfo = {}
    subinfo['hypregmean'] = np.append(0 + 0.5 * np.log(theta.shape[1]) +
                                      np.log(np.std(theta, 0)), (0, hyp1))
    subinfo['hypregLB'] = np.append(-4 + 0.5 * np.log(theta.shape[1]) +
                                    np.log(np.std(theta, 0)), (-12, hyp2))

    subinfo['hypregUB'] = np.append(4 + 0.5 * np.log(theta.shape[1]) +
                                    np.log(np.std(theta, 0)), (2, -8))
    subinfo['hypregstd'] = (subinfo['hypregUB'] - subinfo['hypregLB']) / 8
    subinfo['hypregstd'][-2] = 2
    subinfo['hypregstd'][-1] = 4
    subinfo['hyp'] = 1 * subinfo['hypregmean']
    subinfo['theta'] = theta
    subinfo['g'] = g
    subinfo['sig2ofconst'] = sig2ofconst

    def scaledlik(hypv):
        hyprs = subinfo['hypregmean'] + hypv * subinfo['hypregstd']
        return __negloglik(hyprs, subinfo)

    def scaledlikgrad(hypv):
        hyprs = subinfo['hypregmean'] + hypv * subinfo['hypregstd']
        return __negloglikgrad(hyprs, subinfo) * subinfo['hypregstd']

    newLB = (subinfo['hypregLB'] - subinfo['hypregmean']) / subinfo['hypregstd']
    newUB = (subinfo['hypregUB'] - subinfo['hypregmean']) / subinfo['hypregstd']

    newhyp0 = (subinfo['hyp'] - subinfo['hypregmean']) / subinfo['hypregstd']
    opval = spo.minimize(scaledlik,
                         newhyp0,
                         method='L-BFGS-B',
                         options={'gtol': 0.1},
                         jac=scaledlikgrad,
                         bounds=spo.Bounds(newLB, newUB))

    hypn = subinfo['hypregmean'] + opval.x * subinfo['hypregstd']

    subinfo['hyp'] = hypn
    subinfo['hypcov'] = subinfo['hyp'][:-1]
    subinfo['nug'] = np.exp(subinfo['hyp'][-1]) / (1 + np.exp(subinfo['hyp'][-1]))

    R = __covmat(theta, theta, subinfo['hypcov'])
    subinfo['R'] = (1 - subinfo['nug']) * R + subinfo['nug'] * np.eye(R.shape[0])

    n = subinfo['R'].shape[0]
    W, V = np.linalg.eigh(subinfo['R'])
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ g
    subinfo['sig2'] = (np.mean(fcenter ** 2) * n + sig2ofconst) / (n + sig2ofconst)
    subinfo['Rinv'] = Vh @ Vh.T
    subinfo['Vh'] = Vh
    subinfo['pw'] = subinfo['Rinv'] @ g
    return subinfo


def __negloglik(hyp, info):
    """Return penalized log likelihood of single dimensional GP model."""
    R0 = __covmat(info['theta'], info['theta'], hyp[:-1])
    nug = np.exp(hyp[-1]) / (1 + np.exp(hyp[-1]))
    R = (1 - nug) * R0 + nug * np.eye(info['theta'].shape[0])

    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['g']
    n = info['g'].shape[0]

    sig2ofconst = info['sig2ofconst']
    sig2hat = (n * np.mean(fcenter ** 2) + sig2ofconst) / (n + sig2ofconst)
    negloglik = 1 / 2 * np.sum(np.log(np.abs(W))) + 1 / 2 * n * np.log(sig2hat)
    negloglik += 0.5 * np.sum(((10 ** (-8) + hyp - info['hypregmean']) /
                               (info['hypregstd'])) ** 2)
    return negloglik


def __negloglikgrad(hyp, info):
    """Return gradient of the penalized log likelihood of single dimensional
    GP model."""
    R0, dR = __covmat(info['theta'], info['theta'], hyp[:-1], True)
    nug = np.exp(hyp[-1]) / (1 + np.exp(hyp[-1]))
    R = (1 - nug) * R0 + nug * np.eye(info['theta'].shape[0])
    dR = (1 - nug) * dR
    dRappend = nug / (1 + np.exp(hyp[-1])) * (-R0 + np.eye(info['theta'].shape[0]))

    dR = np.append(dR, dRappend[:, :, None], axis=2)
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['g']
    n = info['g'].shape[0]

    sig2ofconst = info['sig2ofconst']
    sig2hat = (n * np.mean(fcenter ** 2) + sig2ofconst) / (n + sig2ofconst)
    dnegloglik = np.zeros(dR.shape[2])
    Rinv = Vh @ Vh.T

    for k in range(0, dR.shape[2]):
        dsig2hat = - np.sum((Vh @
                             np.multiply.outer(fcenter, fcenter) @
                             Vh.T) * dR[:, :, k]) / (n + sig2ofconst)
        dnegloglik[k] += 0.5 * n * dsig2hat / sig2hat
        dnegloglik[k] += 0.5 * np.sum(Rinv * dR[:, :, k])

    dnegloglik += (10 ** (-8) +
                   hyp - info['hypregmean']) / ((info['hypregstd']) ** 2)
    return dnegloglik


def predict(predinfo, fitinfo, x, theta, **kwargs):
    r"""
    Finds prediction at theta and x given the dictionary fitinfo.
    This [emulationpredictdocstring] automatically filled by docinfo.py when
    running updatedocs.py

    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where you should place all of your prediction
        information once complete. This dictionary is passed by reference, so
        there is no reason to return anything. Keep only stuff that will be
        used by predict. Key elements are

            - `predinfo['mean']` : `predinfo['mean'][k]` is the mean of the prediction
              at all x at `theta[k]`.
            - `predinfo['var']` : `predinfo['var'][k]` is the variance of the
              prediction at all x at `theta[k]`.
            - `predinfo['cov']` : `predinfo['cov'][k]` is the covariance matrix of the
              prediction at all x at `theta[k]`.
            - `predinfo['covhalf']` : if `A = predinfo['covhalf'][k]` then
              `A.T @ A = predinfo['cov'][k]`.

    fitinfo : dict
        An arbitary dictionary where you placed all your important fitting
        information from the fit function above.

    x : array of objects
        An matrix (vector) of inputs for prediction.

    theta :  array of objects
        An matrix (vector) of parameters to prediction.

    kwargs : dict
        A dictionary containing additional options.
    """
    return_grad = False
    if (kwargs is not None) and ('return_grad' in kwargs.keys()) and \
            (kwargs['return_grad'] is True):
        return_grad = True
    return_covx = True
    if (kwargs is not None) and ('return_covx' in kwargs.keys()) and \
            (kwargs['return_covx'] is False):
        return_covx = False
    infos = fitinfo['emulist']
    predvecs = np.full((len(infos), theta.shape[0]), np.nan)
    predvars = np.full((len(infos), theta.shape[0]), np.nan)

    if return_grad:
        predvecs_gradtheta = np.zeros((theta.shape[0], len(infos),
                                       theta.shape[1]))
        predvars_gradtheta = np.zeros((theta.shape[0], len(infos),
                                       theta.shape[1]))
        drsave = np.array(np.ones(len(infos)), dtype=object)
    if predvecs.ndim < 1.5:
        predvecs = predvecs.reshape((1, -1))
        predvars = predvars.reshape((1, -1))

    rsave = np.array(np.ones(len(infos)), dtype=object)

    # loop over each x index
    for k in fitinfo['buildinds']:
        # covariance matrix between new theta and thetas from fit.
        mask_k = fitinfo['mof'][k]
        if return_grad:
            rsave[k], drsave[k] = __covmat(theta,
                                           fitinfo['theta'][~mask_k],
                                           infos[k]['hypcov'],
                                           return_gradx1=True)
        else:
            rsave[k] = __covmat(theta,
                                fitinfo['theta'][~mask_k],
                                infos[k]['hypcov'])
    # adjusted covariance matrix
        r = (1 - infos[k]['nug']) * np.squeeze(rsave[k])

        try:
            rVh = r @ infos[k]['Vh']
            rVh2 = rVh @ (infos[k]['Vh']).T
        except Exception:
            for i in range(0, len(infos)):
                print((i, k))
            raise ValueError('Something went wrong with fitted components')

        if rVh.ndim < 1.5:
            rVh = rVh.reshape((1, -1))
        if rVh2.ndim < 1.5:
            rVh2 = np.reshape(rVh2, (1, -1))
        predvecs[k] = r @ infos[k]['pw']
        if return_grad:
            drsave_hypind = np.squeeze(drsave[k])
            if drsave_hypind.ndim < 2.5 and theta.shape[1] < 1.5:
                drsave_hypind = np.reshape(drsave_hypind, (*drsave_hypind.shape, 1))
            elif drsave_hypind.ndim < 2.5 and theta.shape[1] > 1.5:
                drsave_hypind = np.reshape(drsave_hypind, (1, *drsave_hypind.shape))

            dr = (1 - infos[k]['nug']) * drsave_hypind
            if dr.ndim == 2:
                drVh = dr.T @ infos[k]['Vh']
                predvecs_gradtheta[:, k, :] = dr.T @ infos[k]['pw']
                predvars_gradtheta[:, k, :] = \
                    -infos[k]['sig2'] * 2 * np.sum(rVh * drVh, 1)
            else:
                drpw = np.squeeze(dr.transpose(0, 2, 1) @ infos[k]['pw'])
                if drpw.ndim < 1.5 and theta.shape[1] < 1.5:
                    drpw = np.reshape(drpw, (-1, 1))
                elif drpw.ndim < 1.5 and theta.shape[1] > 1.5:
                    drpw = np.reshape(drpw, (1, -1))

                predvecs_gradtheta[:, k, :] = (1 - infos[k]['nug']) * drpw
                predvars_gradtheta[:, k, :] = \
                    -(infos[k]['sig2'] * 2) * np.einsum("ij,ijk->ik", rVh2, dr)
        predvars[k] = infos[k]['sig2'] * np.abs(1 - np.sum(rVh ** 2, 1))

    # calculate predictive mean and variance
    predinfo['mean'] = (predvecs.T * fitinfo['scale'] + fitinfo['offset']).T
    predinfo['var'] = (predvars.T * (fitinfo['scale'] ** 2)).T
    predinfo['predvecs'] = predvecs
    predinfo['predvars'] = predvars

    if return_covx:
        CH = np.sqrt(predinfo['var'])

        predinfo['covxhalf'] = np.full((theta.shape[0],
                                        CH.shape[1],
                                        x.shape[0]), np.nan)
        for k in range(len(infos)):
            predinfo['covxhalf'][:, :, k] = np.diag(CH[k])
        predinfo['covxhalf'] = predinfo['covxhalf'].transpose((2, 0, 1))

    return


def predictlpdf(predinfo, f, return_grad=False, addvar=0, **kwargs):
    totvar = addvar + predinfo['extravar']
    rf = ((f.T - predinfo['mean'].T) * (1 / np.sqrt(totvar))).T
    Gf = predinfo['phi'].T * (1 / np.sqrt(totvar))
    Gfrf = Gf @ rf
    Gf2 = Gf @ Gf.T
    likv = np.sum(rf ** 2, 0)
    if return_grad:
        rf2 = -(predinfo['mean_gradtheta'].transpose(2, 1, 0) *
                (1 / np.sqrt(totvar))).transpose(2, 1, 0)
        Gfrf2 = (Gf @ rf2.transpose(1, 0, 2)).transpose(1, 0, 2)
        dlikv = 2 * np.sum(rf2.transpose(2, 1, 0) * rf.transpose(1, 0), 2).T
    for c in range(0, predinfo['predvars'].shape[0]):
        w, v = np.linalg.eig(np.diag(1 / (predinfo['predvars'][c, :])) + Gf2)
        term1 = (v * (1 / w)) @ (v.T @ Gfrf[:, c])

        likv[c] -= Gfrf[:, c].T @ term1
        likv[c] += np.sum(np.log(predinfo['predvars'][c, :]))
        likv[c] += np.sum(np.log(w))
        if return_grad:
            Si = (v * (1 / w)) @ v.T
            grt = (predinfo['predvars_gradtheta'][c, :, :].T / predinfo['predvars'][c, :]).T
            dlikv[c, :] += np.sum(grt, 0)
            grt = (-grt.T / predinfo['predvars'][c, :]).T
            dlikv[c, :] += np.diag(Si) @ grt
            term2 = (term1 / predinfo['predvars'][c, :]) ** 2
            dlikv[c, :] -= 2 * Gfrf2[:, c, :].T @ term1
            dlikv[c, :] -= term2 @ (predinfo['predvars_gradtheta'][c, :, :])
    if return_grad:
        return (-likv / 2).reshape(-1, 1), (-dlikv / 2)
    else:
        return (-likv / 2).reshape(-1, 1)


def __generate_param_str(fitinfo):
    """
    Generate a string to describe any information from the fitted emulator,
    including magnitude of residuals, number of GP components, and a summary
    of GP parameters.
    """
    numpc = len(fitinfo['emulist'])
    gp_lengthscales = np.array([fitinfo['emulist'][k]['hypcov'] for k in range(numpc)])
    gp_nuggets = [fitinfo['emulist'][k]['nug'] for k in range(numpc)]

    param_desc = '\tnumber of GP components:\t{:d}\n' \
                 '\tGP parameters, following Gramacy (ch.5, 2022) notations:\n' \
                 '\t\tlengthscales (in log):\n\t\t\t{:s}\n' \
                 '\t\tnuggets (in log):\t{:s}\n' \
        .format(numpc,
                pformat(gp_lengthscales).replace('\n', '\n\t\t\t'),
                pformat(['{:.3f}'.format(np.log(x)) for x in gp_nuggets])
                )

    fitinfo['param_desc'] = param_desc
    return
