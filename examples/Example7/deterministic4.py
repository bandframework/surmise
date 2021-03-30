#breakpoint()
import numpy as np
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

print('N:', func_eval.shape[0])
print('D:', param_values.shape[1])
print('M:', real_data.shape[0])
print('P:', description.shape[1])

# Get the random sample of 500
rndsample = sample(range(0, 2000), 2000)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]


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

# (No filter) Fit an emulator via 'PCGP'
emulator_1 = emulator(x = x,
                      theta = param_values_rnd,
                      f = func_eval_rnd.T,
                      method = 'PCGPwM') 

# (Filter) Fit an emulator via 'PCGP'
emulator_f_1 = emulator(x = x,
                        theta = par_in,
                        f = func_eval_in.T,
                        method = 'PCGPwM') 

# Define a class for prior of 10 parameters
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
obsvar = np.maximum(0.2*real_data, 5) 

list0 = list(range(9, 12))
list10 = []
for j in range(20):
    list10.extend([item + 9*j for item in list0])
    
list11 = []
list11.extend([item + 192 for item in list10])

list12 = []
list12.extend([item + 192 for item in list11])

list10.extend(list11)
list10.extend(list12)

list2 = []
for j in range(576):
    list2.extend([j])


tr_list = [element for i, element in enumerate(list2) if element not in list10]
 
xc_test = x[list10, :]
xc_tr = x[tr_list, :]
y_tr = real_data[tr_list]
obsvar_tr = obsvar[tr_list]

cal_f = calibrator(emu = emulator_f_1,
                   y = y_tr,
                   x = xc_tr,
                   thetaprior = prior_covid,
                   method = 'mlbayes',
                   yvar = obsvar_tr, 
                   args = {'theta0': np.array([[2.9, 0.6, 4, 4]]), 
                           'numsamp' : 500, 
                           'stepType' : 'normal', 
                           'stepParam' : np.array([0.01, 0.01, 0.01, 0.01])})

plot_pred_interval(cal_f, xc_tr, y_tr)
cal_f_theta = cal_f.theta.rnd(500) 
boxplot_param(cal_f_theta)

#breakpoint()
cal_f_ml = calibrator(emu = emulator_f_1,
                      y = y_tr,
                      x = xc_tr,
                      thetaprior = prior_covid,
                      method = 'mlbayes',
                      yvar = obsvar_tr, 
                      args = {'clf_method': model, 
                              'theta0': np.array([[2.9, 0.6, 4, 4]]), 
                              'numsamp' : 500, 
                              'stepType' : 'normal', 
                              'stepParam' : np.array([0.01, 0.01, 0.01, 0.01])})

plot_pred_interval(cal_f_ml, x, real_data)
cal_f_ml_theta = cal_f_ml.theta.rnd(500) 
boxplot_param(cal_f_ml_theta)