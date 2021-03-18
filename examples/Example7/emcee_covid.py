import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from random import sample
import scipy.stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from surmise.emulation import emulator
from surmise.calibration import calibrator
from visualization_tools import boxplot_param
from visualization_tools import plot_pred_interval
from visualization_tools import plot_model_data
from random import sample

# Define a class for prior of 10 parameters
def log_prior(theta):
    a, b, c, d = theta
    if 1.9 < a < 3.9 and 0.29 < b < 1.4 and 3 < c < 5 and 3 < c < 5:
        return 0.0
    return -np.inf

class prior_covid:
    """ This defines the class instance of priors provided to the method. """
    #sps.uniform.logpdf(theta[:, 0], 3, 4.5)
    def lpdf(theta):
        return (sps.uniform.logpdf(theta[:, 0], 1.9, 2) +
                sps.uniform.logpdf(theta[:, 1], 0.29, 1.11) + 
                sps.uniform.logpdf(theta[:, 2], 3, 2) + 
                sps.uniform.logpdf(theta[:, 3], 3, 2)).reshape((len(theta), 1))


    def rnd(n):
        return np.vstack((sps.uniform.rvs(1.9, 2, size=n),
                          sps.uniform.rvs(0.29, 1.11, size=n),
                          sps.uniform.rvs(3, 2, size=n),
                          sps.uniform.rvs(3, 2, size=n))).T


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

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return np.inf
    
    logpost = lp + log_likelihood(theta, obsvar, emu, y, x)
    if clf_method is not None:
        ml_probability = clf_method.predict_proba(theta)[:, 1]
        ml_logprobability = np.reshape(np.log(ml_probability),
                                       (len(theta), 1))
        logpost += ml_logprobability

    return float(logpost)




# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')

# Get the random sample of 500
rndsample = sample(range(0, 100), 100)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

# Filter out the data
par_in = param_values_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 100, func_eval_rnd[:, 100] < 1000)), :]
func_eval_in = func_eval_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 100,  func_eval_rnd[:, 100] < 1000)), :]

# (Filter) Observe computer model outputs
plot_model_data(description, func_eval_in, real_data, par_in)

x = np.hstack((np.reshape(np.tile(range(192), 3), (576, 1)),
              np.reshape(np.tile(np.array(('tothosp', 'icu', 'totadmiss')), 192), (576, 1))))
x =  np.array(x, dtype='object')

# (Filter) Fit an emulator via 'PCGP'
emulator_f_1 = emulator(x=x,
                        theta=param_values_rnd,
                        f=np.sqrt(func_eval_rnd),
                        method='PCGPwM')

obsvar = np.maximum(0.01*np.sqrt(real_data), 1)

from scipy.optimize import minimize

np.random.seed(42)
nll = lambda *args: log_likelihood(*args)
initial = prior_covid.rnd(1)
soln = minimize(nll, initial, args=(obsvar, emulator_f_1, real_data, x))
a, b, c, d = soln.x

import emcee
pos = soln.x + 1e-4 * np.random.randn(32, 4)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(prior_covid, emulator_f_1, real_data, x, obsvar, None))
sampler.run_mcmc(pos, 1000, progress=True);

flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
print(flat_samples.shape)




mean_pred = np.zeros((len(flat_samples), len(real_data)))
for j in range(len(flat_samples)):
    mean_pred[j, :] = emulator_f_1.predict(x=x, theta=flat_samples[j,:]).mean().reshape((576,))
    

plt.rcParams["font.size"] = "10"
fig, axs = plt.subplots(3, figsize=(8, 12))
dim = int(len(x)/3)
for j in range(3):
    upper = np.percentile(mean_pred[:, j*dim : (j + 1)*dim], 97.5, axis = 0)
    lower = np.percentile(mean_pred[:, j*dim : (j + 1)*dim], 2.5, axis = 0)
    median = np.percentile(mean_pred[:, j*dim : (j + 1)*dim], 50, axis = 0)
    p1 = axs[j].plot(median, color = 'black')
    axs[j].fill_between(range(0, dim), lower, upper, color = 'grey')
    p3 = axs[j].plot(range(0, dim), np.sqrt(real_data)[j*dim : (j + 1)*dim], 'ro' ,markersize = 5, color='red')
    if j == 0:
        axs[j].set_ylabel('COVID-19 Total Hospitalizations')
    elif j == 1:
        axs[j].set_ylabel('COVID-19 ICU Patients')
    elif j == 2:
        axs[j].set_ylabel('COVID-19 Hospital Admissions')
    axs[j].set_xlabel('Time (days)')  

    axs[j].legend([p1[0], p3[0]], ['prediction','observations'])
fig.tight_layout()
fig.subplots_adjust(top=0.9) 
plt.show()
    
#plot_pred_interval(cal_f, x, np.sqrt(real_data))
#cal_f_theta = cal_f.theta.rnd(500)
#boxplot_param(mcmc)    
#initial = prior_covid.rnd(1)
#log_probability(initial, prior_covid, emulator_f_1, real_data, x, obsvar, None)