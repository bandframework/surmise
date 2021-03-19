# comparison via MH
import numpy as np
from random import sample
import scipy.stats as sps
from ml_methods import fit_RandomForest
from surmise.emulation import emulator
from surmise.calibration import calibrator
from visualization_tools import boxplot_param
from visualization_tools import plot_pred_interval
from visualization_tools import plot_model_data
from visualization_tools import plot_pred_errors

# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')

# Get the random sample of 100
rndsample = sample(range(0, 1000), 100)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

# (No Filter) Observe computer model outputs
plot_model_data(description, np.sqrt(func_eval_rnd), np.sqrt(real_data), param_values_rnd)

# Filter out the data
T0 = 100
T1 = 2000
par_in = param_values_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > T0,
                                                 func_eval_rnd[:, 100] < T1)), :]
func_eval_in = func_eval_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > T0,
                                                    func_eval_rnd[:, 100] < T1)), :]

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

# Define a class for prior of 10 parameters
class prior_covid:
    """ This defines the class instance of priors provided to the method. """
    #sps.uniform.logpdf(theta[:, 0], 3, 4.5)
    def lpdf(theta):
        return (sps.beta.logpdf((theta[:, 0]-1.9)/2, 2, 2) +
                sps.beta.logpdf((theta[:, 1]-0.29)/1.11, 2, 2) +
                sps.beta.logpdf((theta[:, 2]-3)/2, 2, 2) +
                sps.beta.logpdf((theta[:, 3]-3)/2, 2, 2)).reshape((len(theta), 1))
    def rnd(n):
        return np.vstack((1.9+2*sps.beta.rvs(2, 2, size=n),
                          0.29+1.11*sps.beta.rvs(2, 2, size=n),
                          3+2*sps.beta.rvs(2, 2, size=n),
                          3+2*sps.beta.rvs(2, 2, size=n))).T
    
# Fit a classification model
classification_model = fit_RandomForest(func_eval, param_values, T0, T1)

# Run calibration with 5 random initial starting points
obsvar = np.maximum(0.01*np.sqrt(real_data_tr), 1)
for i in range(5):
    initial = prior_covid.rnd(1)  
    cal_f = calibrator(emu = emulator_f_PCGPwM,
                       y = np.sqrt(real_data_tr),
                       x = xtr,
                       thetaprior = prior_covid,
                       method = 'mlbayes',
                       yvar = obsvar,
                       args = {'clf_method': classification_model, 
                               'theta0': initial, 
                               'numsamp' : 500, 
                               'stepType' : 'normal', 
                               'stepParam' : np.array([0.1, 0.06, 0.1, 0.1])})
    
    plot_pred_interval(cal_f, xtr, np.sqrt(real_data_tr))
    cal_f_theta = cal_f.theta.rnd(500)
    boxplot_param(cal_f_theta)
    plot_pred_errors(cal_f, xtest, np.sqrt(real_data_test))
    
    cal_f = calibrator(emu = emulator_f_PCGPwM,
                       y = np.sqrt(real_data_tr),
                       x = xtr,
                       thetaprior = prior_covid,
                       method = 'mlbayes',
                       yvar = obsvar,
                       args = {'theta0': initial, 
                               'numsamp' : 500, 
                               'stepType' : 'normal', 
                               'stepParam' : np.array([0.1, 0.06, 0.1, 0.1])})
    
    plot_pred_interval(cal_f, xtr, np.sqrt(real_data_tr))
    cal_f_theta = cal_f.theta.rnd(500)
    boxplot_param(cal_f_theta)
    plot_pred_errors(cal_f, xtest, np.sqrt(real_data_test))
    