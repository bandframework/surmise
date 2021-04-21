# comparison via LMC

import numpy as np
from random import sample
import scipy.stats as sps
from ml_methods import fit_RandomForest
from ml_methods import fit_logisticRegression
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
rndsample = sample(range(0, 1000), 1000) #[i for i in range(3000)] #sample(range(0, 1000), 500)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

# (No Filter) Observe computer model outputs
plot_model_data(description, func_eval_rnd, real_data, param_values_rnd)

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

#pair_scatter(par_in)
#pair_scatter(par_out)
# (Filter) Observe computer model outputs
plot_model_data(description, func_eval_in, real_data, par_in)

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
    
#pair_scatter(prior_covid.rnd(1000))

 
def log_likelihood(theta, obsvar, emu, y, x):
    r"""
    This is a optional docstring for an internal function.
    """

    a, b, c, d = theta
    param = np.array([[a, b, c, d]])

    emupredict = emu.predict(x, param)
    emumean = emupredict.mean()
    emuvar = emupredict.var()
    emucovxhalf = emupredict.covxhalf()
    loglik = np.zeros((emumean.shape[1], 1))

    if np.any(np.abs(emuvar/(10 ** (-4) +
                              (1 + 10**(-4))*np.sum(np.square(emucovxhalf),
                                                    2))) > 1):
        emuoldpredict = emu.predict(x)
        emuoldvar = emuoldpredict.var()
        emuoldcxh = emuoldpredict.covxhalf()
        obsvar += np.mean(np.abs(emuoldvar -
                                  np.sum(np.square(emuoldcxh), 2)), 1)

    # compute loglikelihood for each theta value in theta
    for k in range(0, emumean.shape[1]):
        m0 = emumean[:, k]
        S0 = np.squeeze(emucovxhalf[:, k, :])
        stndresid = (np.squeeze(y) - m0) / np.sqrt(obsvar)
        term1 = np.sum(stndresid ** 2)
        J = (S0.T / np.sqrt(obsvar)).T
        if J.ndim < 1.5:
            J = J[:, None]
            stndresid = stndresid[:, None]
        J2 = J.T @ stndresid
        W, V = np.linalg.eigh(np.eye(J.shape[1]) + J.T @ J)
        if W.shape[0] > 1:
            J3 = V @ np.diag(1/W) @ V.T @ J2
        else:
            J3 = ((V**2)/W) * J2
        term2 = np.sum(J3 * J2)
        residsq = term1 - term2
        loglik[k, 0] = -0.5 * residsq - 0.5 * np.sum(np.log(W))

    return float(loglik)

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
            
            param_f = 1*param
            for i_id in range(0, 4):
                for j_id in range(i_id, 4):
                    param_f = np.concatenate([param_f, np.reshape(param_f[0, i_id] * param_f[0, j_id], (1, 1))], axis = 1)
       
        
            ml_probability = clf_method.predict_proba(param_f)[:, 1]
            #print(ml_probability)
            ml_logprobability = np.log(ml_probability)
            logpost += ml_logprobability

        return float(logpost)
    
# Fit a classification model
classification_model = fit_logisticRegression(func_eval, param_values, T0, T1)

obsvar = np.maximum(0.1*np.sqrt(real_data_tr), 1)

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
#boxplot_param(flat_samples_ml)
plot_pred_errors_emcee(flat_samples_ml, emulator_f_PCGPwM, xtest, np.sqrt(real_data_test), 'Posterior predictions with adjustment factor')


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
#boxplot_param(flat_samples)
plot_pred_errors_emcee(flat_samples, emulator_f_PCGPwM, xtest, np.sqrt(real_data_test), 'Posterior predictions without adjustment factor')



plt.rcParams["font.size"] = "8"
fig, axs = plt.subplots(1, 4, figsize=(14, 4))
paraind = 0
labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
for i in range(4):
    axs[i].boxplot([flat_samples[:, i], flat_samples_ml[:, i]], widths=(0.6, 0.6))
    axs[i].set_title(labels[i], fontsize=16)
fig.tight_layout()
fig.subplots_adjust(bottom=0.05, top=0.95)
plt.show()


# import seaborn as sns
# sns.set_style("whitegrid", {'axes.grid' : False})
# #sns.set(rc={'figure.figsize':(4,3)})
# sns.kdeplot(classification_model.predict_proba(flat_samples_ml)[:, 1], color="black", label='adjustment', linestyle="-", legend = True)
# sns.kdeplot(classification_model.predict_proba(flat_samples)[:, 1], color="black", label='no adjustment', linestyle="--", legend = True)
# plt.xlabel(r'$p(r(\theta)=1|\theta)$', fontsize=16)
# plt.legend()
# plt.show()


ml_sample = np.ones((len(flat_samples_ml), 1)) * np.median(flat_samples_ml, axis=0)
ml_sample[:, 0] = flat_samples_ml[:, 0]
for i in range(0, 4):
    for j in range(i, 4):
        ml_sample = np.concatenate([ml_sample, np.reshape(ml_sample[:, i] * ml_sample[:, j], (len(ml_sample), 1))], axis = 1)

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})


plt.rcParams["font.size"] = "8"
fig, axs = plt.subplots(2, 4, figsize=(14, 6))
paraind = 0
labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']

for idi in range(4):
    ml_sample = np.ones((len(flat_samples_ml), 1)) * np.median(flat_samples_ml, axis=0)
    ml_sample[:, idi] = flat_samples_ml[:, idi]
    
    
    lik = np.zeros((len(flat_samples)))
    for i in range(len(flat_samples_ml)):
        lik[i] = log_likelihood(ml_sample[i,:], obsvar, emulator_f_PCGPwM, np.sqrt(real_data_tr), xtr)
    
    mladj = np.max(lik)
    lik = np.max(lik)/lik #np.exp(lik)/np.max(np.exp(lik))
    
    for i in range(0, 4):
        for j in range(i, 4):
            ml_sample = np.concatenate([ml_sample, np.reshape(ml_sample[:, i] * ml_sample[:, j], (len(ml_sample), 1))], axis = 1)
        
    xp = ml_sample[:, idi]
    sort_id = np.argsort(xp)
    yp = classification_model.predict_proba(ml_sample)[:, 1]
    axs[0, idi].plot(xp[sort_id], yp[sort_id], 'k')
    axs[1, idi].plot(xp[sort_id], lik[sort_id], 'k')
    axs[0, idi].tick_params(axis='x', labelsize= 10)
    axs[1, idi].tick_params(axis='x', labelsize= 10)
    axs[0, idi].tick_params(axis='y', labelsize= 10)
    axs[1, idi].tick_params(axis='y', labelsize= 10)
    axs[0, 0].set_ylabel('acceptance probability', fontsize=12)
    axs[1, 0].set_ylabel('conditional likelihood', fontsize=12)

    axs[0, idi].set_title(label=labels[idi], fontsize=16)

plt.show()


#mladj = np.max(yp)
#ypadj = np.log(np.exp(yp - mladj)) + mladj


