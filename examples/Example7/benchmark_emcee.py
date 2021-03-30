# comparison via LMC

import numpy as np
from random import sample
import scipy.stats as sps
from ml_methods import fit_RandomForest
from surmise.emulation import emulator
from visualization_tools import boxplot_param
from visualization_tools import plot_pred_interval_emce
from visualization_tools import plot_model_data
from visualization_tools import plot_pred_errors_emcee
from visualization_tools import pair_scatter

import matplotlib.pyplot as plt
import emcee
# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')

# Get the random sample of 100
rndsample = sample(range(0, 1000), 500)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

# (No Filter) Observe computer model outputs
plot_model_data(description, np.sqrt(func_eval_rnd), np.sqrt(real_data), param_values_rnd)

# Filter out the data
T0 = 100
T1 = 2000
par_in = param_values_rnd[np.logical_and.reduce((func_eval_rnd[:, 25] < 350,
                                                 func_eval_rnd[:, 100] > T0,
                                                 func_eval_rnd[:, 100] < T1)), :]
par_out = param_values_rnd[np.logical_or.reduce((func_eval_rnd[:, 25] > 350,
                                                 func_eval_rnd[:, 100] < T0,
                                                 func_eval_rnd[:, 100] > T1)), :]
func_eval_in = func_eval_rnd[np.logical_and.reduce((func_eval_rnd[:, 25] < 350,
                                                    func_eval_rnd[:, 100] > T0,
                                                    func_eval_rnd[:, 100] < T1)), :]

pair_scatter(par_in)
pair_scatter(par_out)
# (Filter) Observe computer model outputs
plot_model_data(description, np.sqrt(func_eval_in), np.sqrt(real_data), par_in)

# Get the x values 
keeptimepoints = np.arange(10, description.shape[0], step=5)
#keeptimepoints = np.concatenate((np.arange(0, 150), np.arange(0, 150) + 192, np.arange(0, 150) + 2*192))

func_eval_in_tr = func_eval_in[:, keeptimepoints]
real_data_tr = real_data[keeptimepoints]
real_data_test = np.delete(real_data, keeptimepoints, axis=0) 
x = description
xtr = description[keeptimepoints, :]
xtest = np.delete(description, keeptimepoints, axis=0)

# (Filter) Fit an emulator via 'PCGP'
emulator_f_PCGPwM = emulator(x=x,
                             theta=par_in,
                             f=(func_eval_in)**(0.5),
                             method='PCGPwM')

class prior_covid:
    """ This defines the class instance of priors provided to the method. """
    #sps.uniform.logpdf(theta[:, 0], 3, 4.5)
    def lpdf(theta):
        return (sps.beta.logpdf((theta[:, 0]-1)/4, 2, 2) +
                sps.beta.logpdf((theta[:, 1]-0.1)/4.9, 2, 2) +
                sps.beta.logpdf((theta[:, 2]-1)/6, 2, 2) +
                sps.beta.logpdf((theta[:, 3]-1)/6, 2, 2)).reshape((len(theta), 1))
    def rnd(n):
        return np.vstack((1+4*sps.beta.rvs(2, 2, size=n),
                          0.1+4.9*sps.beta.rvs(2, 2, size=n),
                          1+6*sps.beta.rvs(2, 2, size=n),
                          1+6*sps.beta.rvs(2, 2, size=n))).T

# class prior_covid:
#     """ This defines the class instance of priors provided to the method. """
#     #sps.uniform.logpdf(theta[:, 0], 3, 4.5)
#     def lpdf(theta):
#         return (sps.beta.logpdf((theta[:, 0]-1.9)/2, 2, 2) +
#                 sps.beta.logpdf((theta[:, 1]-0.1)/4.9, 2, 2) +
#                 sps.beta.logpdf((theta[:, 2]-3)/2, 2, 2) +
#                 sps.beta.logpdf((theta[:, 3]-3)/2, 2, 2)).reshape((len(theta), 1))
#     def rnd(n):
#         return np.vstack((1.9+2*sps.beta.rvs(2, 2, size=n),
#                           0.1+4.9*sps.beta.rvs(2, 2, size=n),
#                           3+2*sps.beta.rvs(2, 2, size=n),
#                           3+2*sps.beta.rvs(2, 2, size=n))).T
    
# class prior_covid:
#     """ This defines the class instance of priors provided to the method. """
#     #sps.uniform.logpdf(theta[:, 0], 3, 4.5)
#     def lpdf(theta):
#         return (sps.triang.logpdf(theta[:, 0], c=(2.9-1.9)/2, loc=1.9, scale=2) +
#                 sps.triang.logpdf(theta[:, 1], c=(1-0.1)/4.9, loc=0.1, scale=4.9) +
#                 sps.triang.logpdf(theta[:, 2], c=(4-3)/2, loc=3, scale=2) +
#                 sps.triang.logpdf(theta[:, 3], c=(4-3)/2, loc=3, scale=2)).reshape((len(theta), 1))
#     def rnd(n):
#         return np.vstack((sps.triang.rvs(c=(2.9-1.9)/2, loc=1.9, scale=2, size=n),
#                           sps.triang.rvs(c=(1-0.1)/4.9, loc=0.1, scale=4.9, size=n),
#                           sps.triang.rvs(c=(4-3)/2, loc=3, scale=2, size=n),
#                           sps.triang.rvs(c=(4-3)/2, loc=3, scale=2, size=n))).T
    
pair_scatter(prior_covid.rnd(1000))

    
def log_likelihood(theta, obsvar, emu, y, x):
    a, b, c, d = theta
    param = np.array([[a, b, c, d]])
    # Obtain emulator results
    emupredict = emu.predict(x, param)
    emumean = emupredict.mean()

    try:
        emucov = emupredict.covx()
        is_cov = True
    except Exception:
        emucov = emupredict.var()
        is_cov = False

    p = emumean.shape[1]
    n = emumean.shape[0]
    y = y.reshape((n, 1))

    loglikelihood = np.zeros((p, 1))

    for k in range(0, p):
        m0 = emumean[:, k].reshape((n, 1))

        # Compute the covariance matrix
        if is_cov is True:
            s0 = emucov[:, k, :].reshape((n, n))
            CovMat = s0 + np.diag(np.squeeze(obsvar))
        else:
            s0 = emucov[:, k].reshape((n, 1))
            CovMat = np.diag(np.squeeze(s0)) + np.diag(np.squeeze(obsvar))

        # Get the decomposition of covariance matrix
        CovMatEigS, CovMatEigW = np.linalg.eigh(CovMat)

        # Calculate residuals
        resid = m0 - y

        CovMatEigInv = CovMatEigW @ np.diag(1/CovMatEigS) @ CovMatEigW.T
        loglikelihood[k] = float(-0.5 * resid.T @ CovMatEigInv @ resid -
                                 0.5 * np.sum(np.log(CovMatEigS)))

    return float(loglikelihood)

# Define the posterior function
def log_probability(theta, prior_covid, emu, y, x, obsvar, clf_method):
    a, b, c, d = theta
    param = np.array([[a, b, c, d]])
    lp = prior_covid.lpdf(param)
    
    #lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        logpost = lp + log_likelihood(theta, obsvar, emu, y, x)
        if clf_method is not None:
            ml_probability = clf_method.predict_proba(param)[:, 1]
            ml_logprobability = np.log(ml_probability)
            logpost += ml_logprobability

        return float(logpost)
    
# Fit a classification model
classification_model = fit_RandomForest(func_eval, param_values, T0, T1)

obsvar = np.maximum(0.01*np.sqrt(real_data_tr), 1)

# define negative loglikelihood
nll = lambda *args: -log_likelihood(*args)
initial = prior_covid.rnd(50)

# call the sampler
nwalkers, ndim = initial.shape

sampler_ml = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                   args=(prior_covid,
                                         emulator_f_PCGPwM,
                                         np.sqrt(real_data_tr),
                                         xtr, 
                                         obsvar,
                                         classification_model))
sampler_ml.run_mcmc(initial, 1000, progress=True)
flat_samples_ml = sampler_ml.get_chain(discard=500, thin=15, flat=True)
#plot_pred_interval_emce(emulator_f_PCGPwM, flat_samples_ml, xtr, np.sqrt(real_data_tr))
boxplot_param(flat_samples_ml)
plot_pred_errors_emcee(flat_samples_ml, emulator_f_PCGPwM, xtest, np.sqrt(real_data_test))


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                args=(prior_covid,
                                      emulator_f_PCGPwM,
                                      np.sqrt(real_data_tr),
                                      xtr, 
                                      obsvar,
                                      None))
sampler.run_mcmc(initial, 1000, progress=True)
flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
#plot_pred_interval_emce(emulator_f_PCGPwM, flat_samples, xtr, np.sqrt(real_data_tr))
boxplot_param(flat_samples)
plot_pred_errors_emcee(flat_samples, emulator_f_PCGPwM, xtest, np.sqrt(real_data_test))


# (Filter) Fit an emulator via 'PCGP'
emulator_nof_PCGPwM = emulator(x=x,
                               theta=param_values_rnd,
                               f=(func_eval_rnd)**(0.5),
                               method='PCGPwM')

sampler_nof = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(prior_covid,
                                    emulator_nof_PCGPwM,
                                    np.sqrt(real_data_tr),
                                    xtr, 
                                    obsvar,
                                    None))

sampler_nof.run_mcmc(initial, 1000, progress=True)
flat_samples_nof = sampler_nof.get_chain(discard=500, thin=15, flat=True)
#plot_pred_interval_emce(emulator_f_PCGPwM, flat_samples, xtr, np.sqrt(real_data_tr))
boxplot_param(flat_samples_nof)
plot_pred_errors_emcee(flat_samples_nof, emulator_nof_PCGPwM, xtest, np.sqrt(real_data_test))
