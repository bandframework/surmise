# comparison via LMC
import numpy as np
from random import sample
import scipy.stats as sps
from ml_methods import fit_logisticRegression
from surmise.emulation import emulator
from surmise.calibration import calibrator
from visualization_tools import boxplot_compare
from visualization_tools import plot_model_data
from visualization_tools import plot_pred_errors
from additional_visualization import plot_classification_prob
from additional_visualization import plot_loglikelihood
from additional_visualization import log_likelihood
from additional_visualization import plot_adjustedlikelihood

# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')

# Get the random sample of 100
rndsample = sample(range(0, 1000), 1000)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

# (No Filter) Observe computer model outputs
plot_model_data(description, func_eval_rnd, real_data, param_values_rnd)

# Filter out the data
T0 = 50
T1 = 3000
par_in = param_values_rnd[np.logical_and.reduce((func_eval_rnd[:, 25] < 350,
                                                 func_eval_rnd[:, 100] > T0,
                                                 func_eval_rnd[:, 100] < T1)), :]
par_out = param_values_rnd[np.logical_or.reduce((func_eval_rnd[:, 25] > 350,
                                                 func_eval_rnd[:, 100] < T0,
                                                 func_eval_rnd[:, 100] > T1)), :]
func_eval_in = func_eval_rnd[np.logical_and.reduce((func_eval_rnd[:, 25] < 350,
                                                    func_eval_rnd[:, 100] > T0,
                                                    func_eval_rnd[:, 100] < T1)), :]

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

############# THIS PART IS FOR THE PAPER #############
# (No Filter) Fit an emulator via 'PCGP'
emulator_nf_PCGPwM = emulator(x=x,
                              theta=param_values_rnd,
                              f=(func_eval_rnd)**(0.5),
                              method='PCGPwM')

rndsample = sample(range(1000, 2000), 1000)
func_eval_test = func_eval[rndsample, :]
param_values_test = param_values[rndsample, :]

param_values_test = param_values_test[np.logical_and.reduce((func_eval_test[:, 25] < 350,
                                                             func_eval_test[:, 100] > T0,
                                                             func_eval_test[:, 100] < T1)), :]
func_eval_test = func_eval_test[np.logical_and.reduce((func_eval_test[:, 25] < 350,
                                                       func_eval_test[:, 100] > T0,
                                                       func_eval_test[:, 100] < T1)), :]
func_eval_test = (func_eval_test.T)**(0.5)

ftest_mean = emulator_f_PCGPwM.predict(x=x, theta=param_values_test).mean()
nftest_mean = emulator_nf_PCGPwM.predict(x=x, theta=param_values_test).mean()

print(ftest_mean.shape)
print(nftest_mean.shape)
print(func_eval_test.shape)

print(np.sqrt(np.mean((ftest_mean - func_eval_test)**2)))
print(np.sqrt(np.mean((nftest_mean - func_eval_test)**2)))

#############  #############
class prior_covid:
    """ This defines the class instance of priors provided to the method. """
    #sps.uniform.logpdf(theta[:, 0], 3, 4.5)
    def lpdf(theta):
        return (sps.uniform.logpdf(theta[:, 0], 1, 4) +
                sps.uniform.logpdf(theta[:, 1], 0.1, 4.9) +
                sps.uniform.logpdf(theta[:, 2], 1, 6) +
                sps.uniform.logpdf(theta[:, 3], 1, 6)).reshape((len(theta), 1))
    def rnd(n):
        return np.vstack((sps.uniform.rvs(1, 4, size=n),
                          sps.uniform.rvs(0.1, 4.9, size=n),
                          sps.uniform.rvs(1, 6, size=n),
                          sps.uniform.rvs(1, 6, size=n))).T
    
# Fit a classification model
classification_model = fit_logisticRegression(func_eval, param_values, T0, T1)

obsvar = 0.01*real_data_tr

cal_f = calibrator(emu = emulator_f_PCGPwM,
                   y = np.sqrt(real_data_tr),
                   x = xtr,
                   thetaprior = prior_covid,
                   method = 'mlbayeswoodbury',
                   yvar = obsvar,
                   args = {'usedir': True,
                           'sampler':'PTLMC',
                           'maxtemp': 20})

cal_f_theta = cal_f.theta.rnd(500)
plot_pred_errors(cal_f, xtest, np.sqrt(real_data_test))

cal_f_ml = calibrator(emu = emulator_f_PCGPwM,
                   y = np.sqrt(real_data_tr),
                   x = xtr,
                   thetaprior = prior_covid,
                   method = 'mlbayeswoodbury',
                   yvar = obsvar,
                   args = {'usedir': True,
                           'clf_method': classification_model, 
                           'sampler':'PTLMC',
                           'maxtemp': 20})

cal_f_ml_theta = cal_f_ml.theta.rnd(500)
plot_pred_errors(cal_f_ml, xtest, np.sqrt(real_data_test))

boxplot_compare(cal_f_theta, cal_f_ml_theta)

plot_classification_prob(prior_covid, cal_f_ml_theta, classification_model)
plot_loglikelihood(prior_covid, cal_f_ml_theta, obsvar, emulator_f_PCGPwM, real_data_tr, xtr, log_likelihood)
