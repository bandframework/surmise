#This script shows that using/not using derivatives can give very different results for the sampler.


import numpy as np
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

# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')

# Get the random sample of 500
rndsample = sample(range(0, 1000), 100)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

print('N:', func_eval_rnd.shape[0])
print('D:', param_values_rnd.shape[1])
print('M:', real_data.shape[0])
print('P:', description.shape[1])

# (No Filter) Observe computer model outputs
plot_model_data(description, func_eval_rnd, real_data, param_values_rnd)

# Filter out the data
#print(sps.describe(param_values))
par_in = param_values_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 100, func_eval_rnd[:, 100] < 1000)), :]
func_eval_in = func_eval_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 100,  func_eval_rnd[:, 100] < 1000)), :]

# (Filter) Observe computer model outputs
plot_model_data(description, func_eval_in, real_data, par_in)

x = np.hstack((np.reshape(np.tile(range(192), 3), (576, 1)),
              np.reshape(np.tile(np.array(('tothosp', 'icu', 'totadmiss')), 192), (576, 1))))
x =  np.array(x, dtype='object')

keeptimepoints = np.arange(10,x.shape[0],step=5)
func_eval_rnd = func_eval_rnd[:,keeptimepoints]
real_data  = real_data[keeptimepoints]
x  = x[keeptimepoints,:]


# (Filter) Fit an emulator via 'PCGPwM'
emulator_f_PCGPwM = emulator(x=x,
                             theta=param_values_rnd,
                             f=(func_eval_rnd) ** (0.5),
                             method='PCGPwM')
print("built the emulator")

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

##### ##### ##### ##### #####
# Run a classification model
y = np.zeros(len(func_eval))
y[np.logical_and.reduce((func_eval[:, 100] > 100, func_eval[:, 100] < 1000))] = 1

# Create a balanced data set
X_0 = param_values[y == 0][0:1000]
X_0test = param_values[y == 0][1000:2000]
y_0 = y[y == 0][0:1000]
y_0test = y[y == 0][1000:2000]

X_1 = param_values[y == 1][0:1000]
X_1test = param_values[y == 1][1000:2000]
y_1 = y[y == 1][0:1000]
y_1test = y[y == 1][1000:2000]


X = np.concatenate((X_0, X_1), axis=0)
y = np.concatenate((y_0, y_1), axis=0)

Xtest = np.concatenate((X_0test, X_1test), axis=0)
ytest = np.concatenate((y_0test, y_1test), axis=0)

# Fit the classification model
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X, y)

#Training accuracy
print(model.score(X, y))
print(confusion_matrix(y, model.predict(X)))

#Test accuracy
print(confusion_matrix(ytest, model.predict(Xtest)))
print(model.score(Xtest, ytest))
##### ##### ##### ##### #####
obsvar = np.maximum(0.1*np.sqrt(real_data), 0.1)


cal_f = calibrator(emu = emulator_f_PCGPwM,
                   y = (real_data) ** (0.5),
                   x = x,
                   thetaprior = prior_covid,
                   method = 'directbayeswoodbury',
                   yvar = obsvar,
                   args = {'usedir': False})
plot_pred_interval(cal_f, x, (real_data) ** 0.5)
cal_f_theta = cal_f.theta.rnd(500)
boxplot_param(cal_f_theta)
print('no derivative quantiles of logpdf: 5%,15%,...95%')

print(np.quantile(cal_f.info['lpdfapproxnorm_un'],np.arange(0, 9)*0.1+0.05))


cal_f = calibrator(emu = emulator_f_PCGPwM,
                   y = (real_data) ** (0.5),
                   x = x,
                   thetaprior = prior_covid,
                   method = 'directbayeswoodbury',
                   yvar = obsvar,
                   args = {'usedir': True})

plot_pred_interval(cal_f, x, (real_data) ** 0.5)
cal_f_theta = cal_f.theta.rnd(500)
boxplot_param(cal_f_theta)

print('derivative quantiles of logpdf: 5\%,15\%,...95\%')
print(np.quantile(cal_f.info['lpdfapproxnorm_un'],np.arange(0, 9)*0.1+0.05))

# cal_f = calibrator(emu = emulator_f_1,
#                    y = np.sqrt(real_data),
#                    x = x,
#                    thetaprior = prior_covid,
#                    method = 'mlbayes',
#                    yvar = obsvar,
#                    args = {'clf_method': model,
#                            'sampler':'LMC'})
#
# plot_pred_interval(cal_f, x, np.sqrt(real_data))
# cal_f_theta = cal_f.theta.rnd(500)
# boxplot_param(cal_f_theta)



# cal_f = calibrator(emu = emulator_f_1,
#                    y = np.sqrt(real_data),
#                    x = x,
#                    thetaprior = prior_covid,
#                    method = 'directbayeswoodbury',
#                    yvar = obsvar)

# plot_pred_interval(cal_f, x, np.sqrt(real_data))
# cal_f_theta = cal_f.theta.rnd(500)
# boxplot_param(cal_f_theta)


# cal_f = calibrator(emu = emulator_f_1,
#                    y = np.sqrt(real_data),
#                    x = x,
#                    thetaprior = prior_covid,
#                    method = 'mlbayes',
#                    yvar = obsvar)

# plot_pred_interval(cal_f, x, np.sqrt(real_data))
# cal_f_theta = cal_f.theta.rnd(500)
# boxplot_param(cal_f_theta)