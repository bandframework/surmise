"""PCGPwM method - PCGP with Missingness (Chan et al., Technometrics 2024), an
extension to PCGP (Higdon et al., JASA 2008). """

import numpy as np
import scipy.optimize as spo
import scipy.linalg as spla
import copy
from surmise.emulationsupport.matern_covmat import covmat as __covmat
from pprint import pformat


def fit(fitinfo, x, theta, f, epsilonPC=0.001, epsilonImpute=10e-6,
        lognugmean=-10, lognugLB=-20, varconstant=None, dampalpha=0.3, eta=10,
        standardpcinfo=None, verbose=0, **kwargs):
    '''
    The purpose of fit is to take information and plug all of our fit
    information into fitinfo, which is a python dictionary.

    .. note::
       This method is summarized in (Chan et al., Technometrics 2024) and is a
       modification of the method proposed by (Higdon et al., JASA 2008).
       Refer to :py:func:`PCGP` for additional details.

    Prior to performing the PCGP method (Higdon et al., JASA 2008), the PCGPwM method
    checks for missingness in `f` and provides imputations for the missing values
    before conducting the PCGP method.  The method adds approximate variance at
    each points requiring imputation.

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
    epsilonImpute : scalar
        A parameter to ensure covariance nonsingularity. The default is 10e-6.
    lognugmean : scalar
        A parameter to control the log of the nugget used in fitting the GPs.
        The suggested range for lognugmean is (-12, -4).  The nugget is estimated,
        and this parameter is used to guide the estimation.
    lognugLB : scalar
        A parameter to control the lower bound of the log of the nugget. The
        suggested range for lognugLB is (-24, -12).
    varconstant : scalar
        A multiplying constant to control the inflation (deflation) of additional
        variances if missing values are present. The default is None, for which the
        parameter will be optimized. A general working range is (np.exp(-4), np.exp(4)).
    dampalpha : scalar
        A parameter to control the rate of increase of variance as amount of missing
        values increases.  The default is 0.3, otherwise an appropriate range is (0, 0.5).
        Values larger than 0.5 are permitted but it leads to poor empirical performance.
    eta : scalar
        A parameter as an upper bound for the additional variance term.  Default is 10.
    standardpcinfo : dict
        A dictionary user supplies that contains information for standardization of `f`,
        in the following format, such that fs = (f - offset) / scale, U are the
        orthogonal basis vectors, and S are the singular values from SVD of `fs`.
        The entry extravar contains the average squared residual for each column (x).
            {'offset': offset,
             'scale': scale,
             'fs': fs,
             'extravar': extravar,
             'U': U,  # optional
             'S': S  # optional
             }

    verbose : scalar
        A parameter to suppress in-method console output.  Use 0 to suppress output,
        use 1 to show output.


    kwargs : dict, optional
        A dictionary containing options. The default is None.

    Returns
    -------
    None.

    '''
    f = f.T
    # Check for missing or failed values
    if not np.all(np.isfinite(f)):
        fitinfo['mof'] = np.logical_not(np.isfinite(f))
        fitinfo['mofrows'] = np.where(np.any(fitinfo['mof'] > 0.5, 1))[0]
    else:
        fitinfo['mof'] = None
        fitinfo['mofrows'] = None

    fitinfo['epsilonImpute'] = epsilonImpute
    fitinfo['epsilonPC'] = epsilonPC
    hyp1 = lognugmean
    hyp2 = lognugLB
    hypvarconst = np.log(varconstant) if varconstant is not None else None

    fitinfo['dampalpha'] = dampalpha
    fitinfo['eta'] = eta

    fitinfo['theta'] = theta
    fitinfo['f'] = f
    fitinfo['x'] = x

    # Standardize the function evaluations f
    if standardpcinfo is None:
        __standardizef(fitinfo)
    else:
        __verify_pcinfo(standardpcinfo, f)
        fitinfo['standardpcinfo'] = standardpcinfo

    # Construct principal components
    __PCs(fitinfo)
    numpcs = fitinfo['pc'].shape[1]

    if verbose > 0:
        print(fitinfo['method'], 'considering ', numpcs, 'PCs')

    # Fit emulators for all PCs
    emulist = __fitGPs(fitinfo, theta, numpcs, hyp1, hyp2, hypvarconst)
    fitinfo['varc_status'] = 'fixed' if varconstant is not None else 'optimized'
    fitinfo['logvarc'] = np.array([emulist[i]['hypvarconst'] for i in range(numpcs)])
    fitinfo['pcstdvar'] = np.exp(fitinfo['logvarc']) * fitinfo['unscaled_pcstdvar']
    fitinfo['emulist'] = emulist

    __generate_param_str(fitinfo)

    return


def predict(predinfo, fitinfo, x, theta, **kwargs):
    r"""
    Finds prediction at theta and x given the dictionary fitinfo.
    This [emulationpredictdocstring] automatically filled by docinfo.py when
    running updatedocs.py

    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where you should place all of your prediction
        information once complete. This dictionary is pass by reference, so
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
        A matrix (vector) of inputs for prediction.

    theta :  array of objects
        A matrix (vector) of parameters to prediction.

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
    predvecs = np.zeros((theta.shape[0], len(infos)))
    predvars = np.zeros((theta.shape[0], len(infos)))

    if return_grad:
        predvecs_gradtheta = np.zeros((theta.shape[0], len(infos),
                                       theta.shape[1]))
        predvars_gradtheta = np.zeros((theta.shape[0], len(infos),
                                       theta.shape[1]))
        drsave = np.array(np.ones(len(infos)), dtype=object)
    if predvecs.ndim < 1.5:
        predvecs = predvecs.reshape((1, -1))
        predvars = predvars.reshape((1, -1))
    try:
        if x is None or np.all(np.equal(x, fitinfo['x'])) or \
                np.allclose(x, fitinfo['x']):
            xind = np.arange(0, x.shape[0])
            xnewind = np.arange(0, x.shape[0])
        else:
            raise ValueError('Currently x should be the same as x used in fitting.')
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

    rsave = np.array(np.ones(len(infos)), dtype=object)

    # loop over principal components
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            # covariance matrix between new theta and thetas from fit.
            if return_grad:
                rsave[k], drsave[k] = __covmat(theta,
                                               fitinfo['theta'],
                                               infos[k]['hypcov'],
                                               return_gradx1=True)
            else:
                rsave[k] = __covmat(theta,
                                    fitinfo['theta'],
                                    infos[k]['hypcov'])
        # adjusted covariance matrix
        r = (1 - infos[k]['nug']) * np.squeeze(rsave[infos[k]['hypind']])

        try:
            rVh = r @ infos[k]['Vh']
            rVh2 = rVh @ (infos[k]['Vh']).T
        except Exception:
            for i in range(0, len(infos)):
                print((i, infos[i]['hypind']))
            raise ValueError('Something went wrong with fitted components')

        if rVh.ndim < 1.5:
            rVh = rVh.reshape((1, -1))
        if rVh2.ndim < 1.5:
            rVh2 = np.reshape(rVh2, (1, -1))
        predvecs[:, k] = r @ infos[k]['pw']
        if return_grad:
            drsave_hypind = np.squeeze(drsave[infos[k]['hypind']])
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
        predvars[:, k] = infos[k]['sig2'] * np.abs(1 - np.sum(rVh ** 2, 1))

    # calculate predictive mean and variance
    predinfo['mean'] = np.full((x.shape[0], theta.shape[0]), np.nan)
    predinfo['var'] = np.full((x.shape[0], theta.shape[0]), np.nan)
    pctscale = (fitinfo['pcti'].T * fitinfo['standardpcinfo']['scale']).T
    # pctscale = (fitinfo['pct'].T * fitinfo['standardpcinfo']['scale']).T
    predinfo['mean'][xnewind, :] = ((predvecs @ pctscale[xind, :].T) +
                                    fitinfo['standardpcinfo']['offset'][xind]).T
    predinfo['var'][xnewind, :] = ((fitinfo['standardpcinfo']['extravar'][xind] +
                                    predvars @ (pctscale[xind, :] ** 2).T)).T

    predinfo['extravar'] = 1 * fitinfo['standardpcinfo']['extravar'][xind]
    predinfo['predvars'] = 1 * predvars
    predinfo['predvecs'] = 1 * predvecs
    predinfo['phi'] = 1 * pctscale[xind, :]
    # record if covx and grad is called
    predinfo['return_covx'] = return_covx
    predinfo['return_grad'] = return_grad

    if return_covx:
        CH = (np.sqrt(predvars)[:, :, None] * (pctscale[xind, :].T)[None, :, :])

        predinfo['covxhalf'] = np.full((theta.shape[0],
                                        CH.shape[1],
                                        x.shape[0]), np.nan)
        predinfo['covxhalf'][:, :, xnewind] = CH
        predinfo['covxhalf'] = predinfo['covxhalf'].transpose((2, 0, 1))

    if return_grad:
        predinfo['mean_gradtheta'] = np.full((x.shape[0],
                                              theta.shape[0],
                                              theta.shape[1]), np.nan)
        predinfo['mean_gradtheta'][xnewind, :, :] = \
            ((predvecs_gradtheta.transpose(0, 2, 1) @
              pctscale[xind, :].T)).transpose((2, 0, 1))
        predinfo['predvars_gradtheta'] = 1 * predvars_gradtheta
        predinfo['predvecs_gradtheta'] = 1 * predvecs_gradtheta

        if return_covx:

            dsqrtpredvars = 0.5 * (predvars_gradtheta.transpose(2, 0, 1) /
                                   np.sqrt(predvars)).transpose(1, 2, 0)

            if np.allclose(xnewind, xind):
                predinfo['covxhalf_gradtheta'] = \
                    (dsqrtpredvars.transpose(2, 0, 1)[:, :, :, None] *
                     (pctscale[xind, :].T)[None, :, :]).transpose(3, 1, 2, 0)
            else:
                # raise valueerror as xnewind and xind should not differ
                raise ValueError('Check shuffling of x indices within predictions.')
                # predinfo['covxhalf_gradtheta'] = np.full((x.shape[0],
                #                                           theta.shape[0],
                #                                           CH.shape[1],
                #                                           theta.shape[1]), np.nan)
                # predinfo['covxhalf_gradtheta'][xnewind] = \
                #     (dsqrtpredvars.transpose(2, 0, 1)[:, :, :, None] *
                #      (pctscale[xind, :].T)[None, :, :]).transpose(3, 1, 2, 0)
    return


def predictlpdf(predinfo, f, addvar=0, **kwargs):
    return_grad = predinfo['return_grad']

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


def supplementtheta(fitinfo, size, theta, thetachoices, choicecosts, cal,
                    **kwargs):
    r'''
    Suggests next parameters and obviates pending parameters for value
    retrieval of `f`.

    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you placed all your important fitting
        information from the fit function above.
    size : integer
        The number of new thetas the user wants.
    theta : array
        An array of theta values where you want to predict.
    thetachoices : array
        An array of thetas to choose from.
    choicecosts : array
        The computation cost of each theta choice given to you.
    cal : instance of emulator class
        An emulator class instance as defined in calibration.
        This will not always be provided.
    **kwargs : dict
        A dictionary containing additional options.  Specific arguments:
            - `'pending'`: a matrix (sized like `f`) to indicate pending value retrieval of `f`
            - `'costpending'`: the cost to obviate pending thetas
            - `'includepending'`: boolean to include pending values for obviation considerations
        Example usage: `kwargs = {'includepending': True, 'costpending': 0.01+0.99*np.mean(pending,0),
                                  'pending': pending}`.

    Returns
    ----------
    Note that we should have `theta.shape[0] * x.shape[0] < size`.
    theta : array
        Suggested parameters for further value retrievals of `f`.
    info : dict
        A dictionary to contain selection and obviation information. Contains arguments:
            - `'crit'`: criteria associated with selected thetas
            - `'obviatesugg'`: indices in `pending` for suggested obviations
    '''
    pending = None
    if ('pending' in kwargs.keys()):
        pending = kwargs['pending'].T
        pendvar = __getnewvar(fitinfo, pending)

    if ('includepending' in kwargs.keys()) and (kwargs['includepending'] is True):
        includepending = True
        if ('costpending' in kwargs.keys()):
            costpending = kwargs['costpending'] * \
                          np.ones(fitinfo['theta'].shape[0])
        else:
            costpending = np.mean(choicecosts) * \
                          np.ones(fitinfo['theta'].shape[0])
    else:
        includepending = False
    if theta is None:
        raise ValueError('this method is designed to take in the '
                         'theta values.')

    infos = copy.copy(fitinfo['emulist'])
    thetaold = copy.copy(fitinfo['theta'])
    varpca = copy.copy(fitinfo['pcstdvar'])
    thetaposs = thetachoices

    rsave = np.array(np.ones(len(infos)), dtype=object)
    rposssave = np.array(np.ones(len(infos)), dtype=object)
    rnewsave = np.array(np.ones(len(infos)), dtype=object)
    R = np.array(np.ones(len(infos)), dtype=object)

    crit = np.zeros(thetaposs.shape[0])
    weightma = np.mean(fitinfo['pct'] ** 2, 0)

    # covariance matrices between new thetas, thetachoices, and thetas in fit.
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            rsave[k] = (1 - infos[k]['nug']) * __covmat(theta, thetaold,
                                                        infos[k]['hypcov'])
            rposssave[k] = (1 - infos[k]['nug']) * __covmat(thetaposs, thetaold,
                                                            infos[k]['hypcov'])
            rnewsave[k] = (1 - infos[k]['nug']) * __covmat(thetaposs, theta,
                                                           infos[k]['hypcov'])
            R[k] = __covmat(thetaold, thetaold, infos[k]['hypcov'])
            R[k] = (1 - infos[k]['nug']) * R[k] + np.eye(R[k].shape[0]) * \
                infos[k]['nug']

    critsave = np.zeros(thetaposs.shape[0])
    critcount = np.zeros((crit.shape[0], len(infos)))

    if pending is None:
        varpcause = 1 * varpca
    else:
        varpcause = 1 * pendvar

    # calculation of selection criterion
    thetachoicesave = np.zeros((size, fitinfo['theta'].shape[1]))
    for j in range(0, size):
        critcount = np.zeros((crit.shape[0], len(infos)))
        if thetaposs.shape[0] < 1.5:
            thetaold = np.vstack((thetaold, thetaposs))
            break
        for k in range(0, len(infos)):
            # if infos[k]['hypind'] == k:  # this is to speed things up a bit
            #     Rh = R[infos[k]['hypind']] + np.diag(varpcause[:, k])
            #     p = rnewsave[infos[k]['hypind']]
            #     term1 = np.linalg.solve(Rh, rposssave[infos[k]['hypind']].T)
            #     q = rsave[infos[k]['hypind']] @ term1
            #     r = rposssave[infos[k]['hypind']].T * term1
            #     critcount[:, k] = weightma[k] * np.mean((p.T - q) ** 2, 0) / \
            #         np.abs(1 - np.sum(r, 0))
            # else:
            critcount[:, k] = weightma[k] / weightma[infos[k]['hypind']] * \
                              critcount[:, infos[k]['hypind']]
        crit = np.sum(critcount, 1)
        jstar = np.argmax(crit / choicecosts)
        critsave[j] = crit[jstar] / choicecosts[jstar]
        thetaold = np.vstack((thetaold, thetaposs[jstar]))
        thetachoicesave[j] = thetaposs[jstar]
        thetaposs = np.delete(thetaposs, jstar, 0)
        for k in range(0, len(infos)):
            if infos[k]['hypind'] == k:
                R[k] = np.vstack((R[k], rposssave[k][jstar, :]))
                R[k] = np.vstack((R[k].T,
                                  np.append(rposssave[k][jstar, :], 1))).T
                newr = (1 - infos[k]['nug']) * \
                    __covmat(thetaposs, thetaold[-1, :], infos[k]['hypcov'])
                rposssave[k] = np.delete(rposssave[k], jstar, 0)
                rposssave[k] = np.hstack((rposssave[k], newr))
                rsave[k] = np.hstack((rsave[k],
                                      rnewsave[k][jstar, :][:, None]))
                rnewsave[k] = np.delete(rnewsave[k], jstar, 0)
        crit = np.delete(crit, jstar)
        critcount = np.delete(critcount, jstar, 0)
        choicecosts = np.delete(choicecosts, jstar)
        varpcause = np.vstack((varpcause, 0 * varpca[0, :]))
        varpca = np.vstack((varpca, 0 * varpca[0, :]))

        for k in range(0, len(infos)):
            if infos[k]['hypind'] == k:
                rsave[k] = (1 - infos[k]['nug']) * __covmat(theta, thetaold,
                                                            infos[k]['hypcov'])
                R[k] = __covmat(thetaold, thetaold, infos[k]['hypcov'])
                R[k] = (1 - infos[k]['nug']) * R[k] + np.eye(R[k].shape[0]) * \
                    infos[k]['nug']

    # calculation of obviation criterion and suggests obviations.
    info = {}
    info['crit'] = critsave
    if includepending:
        critpend = np.zeros((fitinfo['theta'].shape[0], len(infos)))
        for k in range(0, len(infos)):
            # if infos[k]['hypind'] == k:  # this is to speed things up a bit
            #     Rh = R[infos[k]['hypind']] + np.diag(varpca[:, k])
            #     term1 = np.linalg.solve(Rh, rsave[infos[k]['hypind']].T)
            #     delta = (pendvar[:, k] - varpca[:fitinfo['theta'].shape[0], k])
            #     term3 = np.diag(np.linalg.inv(Rh))[:fitinfo['theta'].shape[0]]
            #     critpend[:, k] = -weightma[k] * delta * \
            #         np.mean((term1[:fitinfo['theta'].shape[0], :] ** 2), 1) / (1 + delta * term3)
            # else:
            critpend[:, k] = weightma[k] / weightma[infos[k]['hypind']] * \
                             critpend[:, infos[k]['hypind']]
        critpend = np.sum(critpend, 1)
        info['obviatesugg'] = np.where((np.mean(critsave[:size]) >
                                       critpend / costpending) > 0.5)[0]
    return thetachoicesave, info


def __standardizef(fitinfo, offset=None, scale=None):
    r'''Standardizes f by creating offset, scale and fs.  When appropriate,
    imputes values for `f`.'''
    # Extracting from input dictionary
    f = fitinfo['f']
    mof = fitinfo['mof']
    mofrows = fitinfo['mofrows']
    epsilonPC = fitinfo['epsilonPC']

    if (offset is not None) and (scale is not None):
        if offset.shape[0] == f.shape[1] and scale.shape[0] == f.shape[1]:
            if np.any(np.nanmean(np.abs(f - offset) / scale, 1) > 4):
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
            scale[k] = np.nanstd(f[:, k]) / np.sqrt(1-np.isnan(f[:, k]).mean())
            if scale[k] == 0:
                raise ValueError("You have a row that is non-varying.")

    fs = np.zeros(f.shape)
    if mof is None:
        fs = (f - offset) / scale
    else:
        epsilonImpute = fitinfo['epsilonImpute']
        # Imputes missing values
        for k in range(0, f.shape[1]):
            fs[:, k] = (f[:, k] - offset[k]) / scale[k]
            if np.sum(mof[:, k]) > 0:
                a = np.empty((np.sum(mof[:, k]),))
                a[::2] = 2
                a[1::2] = -2
                fs[np.where(mof[:, k])[0], k] = (a - np.mean(a))

        for iters in range(0, 40):
            U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
            Sp = S ** 2 - epsilonPC
            Up = U[:, Sp > 0]
            Sp = np.sqrt(Sp[Sp > 0])
            for j in range(0, mofrows.shape[0]):
                rv = mofrows[j]
                wheremof = np.where(mof[rv, :] > 0.5)[0]
                wherenotmof = np.where(mof[rv, :] < 0.5)[0]
                H = Up[wherenotmof, :].T @ Up[wherenotmof, :]
                Amat = epsilonImpute * np.diag(1 / (Sp ** 2)) + H
                J = Up[wherenotmof, :].T @ fs[rv, wherenotmof]
                fs[rv, wheremof] = (Up[wheremof, :] *
                                    ((Sp / np.sqrt(epsilonImpute)) ** 2)) @ \
                                   (J - H @ (spla.solve(Amat, J, assume_a='pos')))

    # Assigning new values to the dictionary
    U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
    Sp = S ** 2 - epsilonPC
    Up = U[:, Sp > 0]

    extravar = np.nanmean((fs - fs @ Up @ Up.T) ** 2, 0) * (scale ** 2)

    standardpcinfo = {'offset': offset,
                      'scale': scale,
                      'fs': fs,
                      'U': U,
                      'S': S,
                      'extravar': extravar
                      }

    fitinfo['standardpcinfo'] = standardpcinfo
    return


def __PCs(fitinfo):
    "Apply PCA to reduce the dimension of `f`."
    # Extracting from input dictionary
    f = fitinfo['f']
    mof = fitinfo['mof']
    mofrows = fitinfo['mofrows']
    epsilonPC = fitinfo['epsilonPC']

    fs = fitinfo['standardpcinfo']['fs']
    if 'U' in fitinfo['standardpcinfo']:
        U = fitinfo['standardpcinfo']['U']
        S = fitinfo['standardpcinfo']['S']
    else:
        U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
    Sp = S ** 2 - epsilonPC
    pct = U[:, Sp > 0]
    pcw = np.sqrt(Sp[Sp > 0])
    pc = fs @ pct
    pcstdvar = np.zeros((f.shape[0], pct.shape[1]))
    if mof is not None:
        epsilonImpute = fitinfo['epsilonImpute']
        for j in range(0, mofrows.shape[0]):
            rv = mofrows[j]
            wherenotmof = np.where(mof[rv, :] < 0.5)[0]
            H = pct[wherenotmof, :].T @ pct[wherenotmof, :]
            Amat = np.diag(epsilonImpute / (pcw ** 2)) + H
            J = pct[wherenotmof, :].T @ fs[rv, wherenotmof]
            pc[rv, :] = (pcw ** 2 / epsilonImpute + 1) * \
                        (J - H @ np.linalg.solve(Amat, J))
            fs[rv, :] = pc[rv, :] @ pct.T
            Qmat = np.diag(epsilonImpute / pcw ** 2) + H
            term3 = np.diag(H) - \
                np.sum(H * spla.solve(Qmat, H, assume_a='pos'), 0)
            pcstdvar[rv, :] = 1 - (pcw ** 2 / epsilonImpute + 1) * term3
    fitinfo['pcw'] = pcw
    fitinfo['pcto'] = 1 * pct
    # pcw contains the singular values from SVD, in complete data pcw
    # scales with np.sqrt(pc.shape[0]), where pc.shape[0] is the number
    # of parameters. With missing data, we approximate the growth to be
    # np.sqrt(np.sum(1-pcstdvar)))
    effn = np.sum(np.clip(1 - pcstdvar, 0, 1))
    fitinfo['pct'] = pct * pcw / np.sqrt(effn)
    fitinfo['pcti'] = pct * (np.sqrt(effn) / pcw)
    # fitinfo['pc'] = pc * (np.sqrt(effn) / pcw)
    fitinfo['pc'] = fs @ fitinfo['pct']
    fitinfo['unscaled_pcstdvar'] = pcstdvar
    return


def __getnewvar(fitinfo, pending):
    "Calculates the variances for entries where there are missing values."
    # Extracting from principal components fit dictionary.
    pct = copy.copy(fitinfo['pcto'])
    pcw = copy.copy(fitinfo['pcw'])
    epsilonImpute = fitinfo['epsilonImpute']

    realfail = np.logical_and(np.logical_not(pending), fitinfo['mof'])
    failrows = np.where(np.any(realfail, 1))[0]
    pcstdvar = np.zeros((pending.shape[0], pct.shape[1]))

    for j in range(0, failrows.shape[0]):
        rv = failrows[j]
        wherenotmof = np.where(realfail[rv, :] < 0.5)[0]
        H = pct[wherenotmof, :].T @ pct[wherenotmof, :]
        Qmat = np.diag(epsilonImpute / pcw ** 2) + H
        term3 = np.diag(H) - np.sum(H * spla.solve(Qmat, H, assume_a='pos'), 0)
        pcstdvar[rv, :] = 1 - (pcw ** 2 / epsilonImpute + 1) * term3
    return np.exp(fitinfo['logvarc']) * pcstdvar


def __fitGPs(fitinfo, theta, numpcs, hyp1, hyp2, varconstant):
    """Fit emulators for all principle components."""
    if 'emulist' in fitinfo.keys():
        hypstarts = np.zeros((numpcs, fitinfo['emulist'][0]['hyp'].shape[0]))
        hypinds = -1 * np.ones(numpcs)
        for pcanum in range(0, min(numpcs, len(fitinfo['emulist']))):
            hypstarts[pcanum, :] = fitinfo['emulist'][pcanum]['hyp']
            hypinds[pcanum] = fitinfo['emulist'][pcanum]['hypind']
    else:
        hypstarts = None
        hypinds = -1 * np.ones(numpcs)

    emulist = [dict() for x in range(0, numpcs)]
    for iters in range(0, 3):
        for pcanum in range(0, numpcs):
            if np.sum(hypinds == np.array(range(0, numpcs))) > 0.5:
                hypwhere = np.where(hypinds == np.array(range(0, numpcs)))[0]
                emulist[pcanum] = __fitGP1d(theta=theta,
                                            g=fitinfo['pc'][:, pcanum],
                                            hyp1=hyp1,
                                            hyp2=hyp2,
                                            hypvarconst=varconstant,
                                            gvar=fitinfo['unscaled_pcstdvar'][:, pcanum],
                                            dampalpha=fitinfo['dampalpha'],
                                            eta=fitinfo['eta'],
                                            hypstarts=hypstarts[hypwhere, :],
                                            hypinds=hypwhere,
                                            sig2ofconst=0.01)
            else:
                emulist[pcanum] = __fitGP1d(theta=theta,
                                            g=fitinfo['pc'][:, pcanum],
                                            hyp1=hyp1,
                                            hyp2=hyp2,
                                            hypvarconst=varconstant,
                                            gvar=fitinfo['unscaled_pcstdvar'][:, pcanum],
                                            dampalpha=fitinfo['dampalpha'],
                                            eta=fitinfo['eta'],
                                            sig2ofconst=0.01)
                hypstarts = np.zeros((numpcs, emulist[pcanum]['hyp'].shape[0]))
            emulist[pcanum]['hypind'] = min(pcanum, emulist[pcanum]['hypind'])
            hypstarts[pcanum, :] = emulist[pcanum]['hyp']
            if emulist[pcanum]['hypind'] < -0.5:
                emulist[pcanum]['hypind'] = 1 * pcanum
            hypinds[pcanum] = 1 * emulist[pcanum]['hypind']
    return emulist


def __fitGP1d(theta, g, hyp1, hyp2, hypvarconst, gvar=None, dampalpha=None, eta=None,
              hypstarts=None, hypinds=None, sig2ofconst=None):
    """Return a fitted model from the emulator model using smart method."""
    hypvarconstmean = 4 if hypvarconst is None else hypvarconst
    hypvarconstLB = -8 if hypvarconst is None else hypvarconst - 0.5
    hypvarconstUB = 8 if hypvarconst is None else hypvarconst + 0.5

    subinfo = {}
    subinfo['hypregmean'] = np.append(0 + 0.5 * np.log(theta.shape[1]) +
                                      np.log(np.std(theta, 0)), (0, hypvarconstmean, hyp1))
    subinfo['hypregLB'] = np.append(-4 + 0.5 * np.log(theta.shape[1]) +
                                    np.log(np.std(theta, 0)), (-12, hypvarconstLB, hyp2))

    subinfo['hypregUB'] = np.append(4 + 0.5 * np.log(theta.shape[1]) +
                                    np.log(np.std(theta, 0)), (2, hypvarconstUB, -8))
    subinfo['hypregstd'] = (subinfo['hypregUB'] - subinfo['hypregLB']) / 8
    subinfo['hypregstd'][-3] = 2
    subinfo['hypregstd'][-1] = 4
    subinfo['hyp'] = 1 * subinfo['hypregmean']
    nhyptrain = np.max(np.min((20 * theta.shape[1], theta.shape[0])))
    if theta.shape[0] > nhyptrain:
        thetac = np.random.choice(theta.shape[0], nhyptrain, replace=False)
    else:
        thetac = range(0, theta.shape[0])
    subinfo['theta'] = theta[thetac, :]
    subinfo['g'] = g[thetac]

    # maxgvar = np.max(gvar)
    # gvar = gvar / ((np.abs(maxgvar*1.001 - gvar)) ** dampalpha)

    gvar = np.minimum(eta, gvar / ((1 - gvar)**dampalpha))

    subinfo['sig2ofconst'] = sig2ofconst
    subinfo['gvar'] = gvar[thetac]
    hypind0 = -1

    L0 = __negloglik(subinfo['hyp'], subinfo)
    if hypstarts is not None:
        L0 = __negloglik(subinfo['hyp'], subinfo)
        for k in range(0, hypstarts.shape[0]):
            L1 = __negloglik(hypstarts[k, :], subinfo)
            if L1 < L0:
                subinfo['hyp'] = hypstarts[k, :]
                L0 = 1 * L1
                hypind0 = hypinds[k]

    if hypind0 > -0.5 and hypstarts.ndim > 1:
        dL = __negloglikgrad(subinfo['hyp'], subinfo)
        scalL = np.std(hypstarts, 0) * hypstarts.shape[0] / \
            (1 + hypstarts.shape[0]) + (1 / (1 + hypstarts.shape[0]) * subinfo['hypregstd'])
        if np.sum((dL * scalL) ** 2) < 1.25 * \
                (subinfo['hyp'].shape[0] + 5 * np.sqrt(subinfo['hyp'].shape[0])):
            skipop = True
        else:
            skipop = False
    else:
        skipop = False

    if not skipop:
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
        likdiff = (L0 - __negloglik(hypn, subinfo))
    else:
        likdiff = 0
    if hypind0 > -0.5 and (2 * likdiff) < 1.25 * \
            (subinfo['hyp'].shape[0] + 5 * np.sqrt(subinfo['hyp'].shape[0])):
        subinfo['hypcov'] = subinfo['hyp'][:-2]
        subinfo['hypvarconst'] = subinfo['hyp'][-2]
        subinfo['hypind'] = hypind0
        subinfo['nug'] = np.exp(subinfo['hyp'][-1]) / (1 + np.exp(subinfo['hyp'][-1]))

        R = __covmat(theta, theta, subinfo['hypcov'])

        subinfo['R'] = (1 - subinfo['nug']) * R + subinfo['nug'] * np.eye(R.shape[0])
        if gvar is not None:
            subinfo['R'] += np.exp(subinfo['hypvarconst'])*np.diag(gvar)

        W, V = np.linalg.eigh(subinfo['R'])
        Vh = V / np.sqrt(np.abs(W))
        fcenter = Vh.T @ g
        subinfo['Vh'] = Vh
        n = subinfo['R'].shape[0]
        subinfo['sig2'] = (np.mean(fcenter ** 2) * n + sig2ofconst) / (n + sig2ofconst)
        subinfo['Rinv'] = V @ np.diag(1 / W) @ V.T
    else:
        subinfo['hyp'] = hypn
        subinfo['hypind'] = -1
        subinfo['hypcov'] = subinfo['hyp'][:-2]
        subinfo['hypvarconst'] = subinfo['hyp'][-2]
        subinfo['nug'] = np.exp(subinfo['hyp'][-1]) / (1 + np.exp(subinfo['hyp'][-1]))

        R = __covmat(theta, theta, subinfo['hypcov'])
        subinfo['R'] = (1 - subinfo['nug']) * R + subinfo['nug'] * np.eye(R.shape[0])
        if gvar is not None:
            subinfo['R'] += np.exp(subinfo['hypvarconst'])*np.diag(gvar)
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
    """Return penalized log likelihood of single demensional GP model."""
    R0 = __covmat(info['theta'], info['theta'], hyp[:-2])
    nug = np.exp(hyp[-1]) / (1 + np.exp(hyp[-1]))
    R = (1 - nug) * R0 + nug * np.eye(info['theta'].shape[0])

    if info['gvar'] is not None:
        R += np.exp(hyp[-2])*np.diag(info['gvar'])
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
    """Return gradient of the penalized log likelihood of single demensional
    GP model."""
    R0, dR = __covmat(info['theta'], info['theta'], hyp[:-2], True)
    nug = np.exp(hyp[-1]) / (1 + np.exp(hyp[-1]))
    R = (1 - nug) * R0 + nug * np.eye(info['theta'].shape[0])
    dR = (1 - nug) * dR
    dRappend2 = nug / (1 + np.exp(hyp[-1])) * (-R0 + np.eye(info['theta'].shape[0]))

    if info['gvar'] is not None:
        R += np.exp(hyp[-2]) * np.diag(info['gvar'])
        dRappend1 = np.exp(hyp[-2]) * np.diag(info['gvar'])
    else:
        dRappend1 = 0 * np.eye(info['theta'].shape[0])

    dR = np.append(dR, dRappend1[:, :, None], axis=2)
    dR = np.append(dR, dRappend2[:, :, None], axis=2)
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


def __verify_pcinfo(pcinfo, f):
    def __fill_pcinfo(pcinfo, f):
        """
        Completes pcinfo dictionary if only the basis vectors Phi (U) is given.
        :return:
        """
        Phi = pcinfo['U']

        offset = f.mean(0)
        scale = np.ones(f.shape[1])
        S = np.ones(Phi.shape[1])
        fs = ((f - offset) / scale)
        extravar = np.mean((fs.T - Phi @ Phi.T @ fs.T) ** 2, 1) * (scale ** 2)

        pcinfo['offset'] = offset
        pcinfo['scale'] = scale
        pcinfo['S'] = S
        pcinfo['fs'] = fs
        pcinfo['extravar'] = extravar
        return

    if 'U' not in pcinfo.keys():
        raise AttributeError('\'U\', the basis vectors must be provided.')
    if len(pcinfo.keys()) == 1:
        __fill_pcinfo(pcinfo, f)
    return


def __generate_param_str(fitinfo):
    """
    Generate a string to describe any information from the fitted emulator,
    including magnitude of residuals, number of GP components, and a summary
    of GP parameters.
    """
    numpc = len(fitinfo['emulist'])
    extravar = fitinfo['standardpcinfo']['extravar']
    gp_scales = 1/(1+np.exp(np.array([fitinfo['emulist'][k]['hyp'][-2] for k in range(numpc)])))
    gp_lengthscales = np.array([fitinfo['emulist'][k]['hyp'][:-3] for k in range(numpc)])
    gp_nuggets = np.array([fitinfo['emulist'][k]['nug'] for k in range(numpc)])

    param_desc = '\taverage emulation residual variance (from principal components):\t{:.3E}\n' \
                 '\tnumber of GP components:\t{:d}\n' \
                 '\tGP parameters, following Gramacy (ch.5, 2022) notations:\n' \
                 '\t\tscales (in log): \t{:s}\n' \
                 '\t\tlengthscales (in log):\n\t\t\t{:s}\n' \
                 '\t\tnuggets (in log):\t{:s}\n' \
        .format(extravar.mean(),
                numpc,
                pformat(['{:.3f}'.format(np.log(x)) for x in gp_scales]),
                pformat(gp_lengthscales).replace('\n', '\n\t\t\t'),
                pformat(['{:.3f}'.format(np.log(x)) for x in gp_nuggets])
                )

    fitinfo['param_desc'] = param_desc
    return
