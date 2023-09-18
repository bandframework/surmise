#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:03:39 2022

@author: ozgesurer
"""
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl


def fit(fitinfo, x, theta, f, epsilon=0.1, standardpcinfo=None, **kwargs):
    prior = kwargs['prior']
    f = f.T
    fitinfo['theta'] = theta
    fitinfo['x'] = x
    fitinfo['f'] = f
    fitinfo['epsilon'] = epsilon

    standardizef(fitinfo)
    PCA(fitinfo, standardpcinfo)
    numpcs = fitinfo['pc_scores'].shape[1]

    # create a dictionary to save the emu info for each PC
    emulist = [dict() for x in range(0, numpcs)]

    # fit a GP for each PC
    for pcanum in range(0, numpcs):
        emulist[pcanum] = emulation_fit(theta, fitinfo['pc_scores'][:, pcanum], prior)

    fitinfo['emulist'] = emulist
    return


def emulation_fit(theta, pcaval, prior):
    prior_min = prior['min']
    prior_max = prior['max']

    ptp = np.array(prior_max) - np.array(prior_min)

    kernel = (1 * krnl.RBF(length_scale=ptp, length_scale_bounds=np.outer(ptp, (1e-3, 1e3))) +
              krnl.WhiteKernel(noise_level=.1, noise_level_bounds=(1e-3, 1e1)))
    GPR = gpr(kernel=kernel, n_restarts_optimizer=50, alpha=0.0000000001)
    GPR.fit(theta, pcaval.reshape(-1, 1))

    print(f'GPR score is {GPR.score(theta, pcaval)} \n')
    return GPR


def standardizef(fitinfo):
    # Scaling the data to be zero mean and unit variance for each observable
    f = fitinfo['f']
    SS = StandardScaler(copy=True)
    fs = SS.fit_transform(f)
    fitinfo['fs'] = fs
    fitinfo['ss'] = SS


def PCA(fitinfo, standardpcinfo):
    fs = fitinfo['fs']
    SS = fitinfo['ss']
    if standardpcinfo is None:
        epsilon = 1 - fitinfo['epsilon']

        u, s, vh = np.linalg.svd(fs, full_matrices=True)

        importance = np.square(s / math.sqrt(u.shape[0] - 1))
        cum_importance = np.cumsum(importance) / np.sum(importance)

        pc_no = [c_id for c_id, c in enumerate(cum_importance) if c > epsilon][0]

        # Scale transformation from PC space to original data space
        inverse_tf_matrix = np.diag(s[0:pc_no]) @ vh[0:pc_no, :] * SS.scale_.reshape(1, -1) / math.sqrt(u.shape[0] - 1)

        print('pc_no:', pc_no)
        pc_scores = u[:, 0:pc_no] * math.sqrt(u.shape[0] - 1)

        fitinfo['pc_scores'] = pc_scores
        fitinfo['inverse_tf'] = inverse_tf_matrix
    else:
        u = standardpcinfo['U']
        s = standardpcinfo['S']
        vh = standardpcinfo['V']
        inverse_tf_matrix = np.diag(s) @ vh * SS.scale_.reshape(1, -1) / math.sqrt(u.shape[0] - 1)
        pc_scores = u * math.sqrt(u.shape[0] - 1)

        fitinfo['pc_scores'] = pc_scores
        fitinfo['inverse_tf'] = inverse_tf_matrix


def predict(predinfo, fitinfo, x, theta, computecov=True, **kwargs):
    d = x.shape[0]

    prediction_val = []
    prediction_sig_val = []
    for row in theta:
        prediction, pred_cov = predict_observables(row, fitinfo)
        prediction_sig_val.append(np.sqrt(np.diagonal(pred_cov)))
        prediction_val.append(prediction)

    prediction_val = np.array(prediction_val).reshape(-1, d)
    prediction_sig_val = np.array(prediction_sig_val).reshape(-1, d)

    predinfo['mean'] = prediction_val.T
    predinfo['var'] = (prediction_sig_val.T) ** 2


def predict_observables(model_parameters, fitinfo):
    """Predicts the observables for any model parameter value using the trained emulators.

    Parameters
    ----------
    Theta_input : Model parameter values. Should be a 1D array of 15 model parameters.

    Return
    ----------
    Mean value and full error covariance matrix of the prediction. """

    emulist = fitinfo['emulist']
    SS = fitinfo['ss']
    inverse_tf_matrix = fitinfo['inverse_tf']
    mean = []
    variance = []
    theta = np.array(model_parameters).flatten()

    theta = np.array(theta).reshape(1, 15)
    npc = len(emulist)

    for i in range(0, npc):
        mn, std = emulist[i].predict(theta, return_std=True)
        mean.append(mn)
        variance.append(std ** 2)

    mean = np.array(mean).reshape(1, -1)
    inverse_transformed_mean = mean @ inverse_tf_matrix + np.array(SS.mean_).reshape(1, -1)
    variance_matrix = np.diag(np.array(variance).flatten())
    A_p = inverse_tf_matrix
    inverse_transformed_variance = np.einsum('ik,kl,lj-> ij', A_p.T, variance_matrix, A_p, optimize=False)

    return inverse_transformed_mean, inverse_transformed_variance
