#breakpoint()
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import scipy.stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from surmise.emulation import emulator
from surmise.calibration import calibrator

# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = 1/np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')
param_values_test = 1/np.loadtxt('param_values_test.csv', delimiter=',')
func_eval_test = np.loadtxt('func_eval_test.csv', delimiter=',')

# Remove the initial 30-days time period from the data
keepinds = np.squeeze(np.where(description[:,0].astype('float') > 30))
real_data = real_data[keepinds]
description = description[keepinds, :]
func_eval = func_eval[:,keepinds]
func_eval_test = func_eval_test[:, keepinds]

print('N:', func_eval.shape[0])
print('D:', param_values.shape[1])
print('M:', real_data.shape[0])
print('P:', description.shape[1])

# Get the random sample of 500
rndsample = sample(range(0, 2000), 2000)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

def boxplot_param(theta):
    plt.rcParams["font.size"] = "16"
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    paraind = 0
    for i in range(2):
        for j in range(5):
            axs[i, j].boxplot(theta[:, paraind])
            paraind += 1

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05, top=0.95)
    plt.show()

def plot_pred_interval(cal):
    pr = cal.predict(x)
    rndm_m = pr.rnd(s = 100)
    plt.rcParams["font.size"] = "10"
    fig, axs = plt.subplots(3, figsize=(8, 12))

    for j in range(3):
        upper = np.percentile(rndm_m[:, j*134 : (j + 1)*134], 97.5, axis = 0)
        lower = np.percentile(rndm_m[:, j*134 : (j + 1)*134], 2.5, axis = 0)
        median = np.percentile(rndm_m[:, j*134 : (j + 1)*134], 50, axis = 0)
        p1 = axs[j].plot(median, color = 'black')
        axs[j].fill_between(range(0, 134), lower, upper, color = 'grey')
        p3 = axs[j].plot(range(0, 134), real_data[j*134 : (j + 1)*134], 'ro' ,markersize = 5, color='red')
        if j == 0:
            axs[j].set_ylabel('COVID-19 Total Hospitalizations')
        elif j == 1:
            axs[j].set_ylabel('COVID-19 Hospital Admissions')
        elif j == 2:
            axs[j].set_ylabel('COVID-19 ICU Patients')
        axs[j].set_xlabel('Time (days)')  
    
        axs[j].legend([p1[0], p3[0]], ['prediction','observations'])
    fig.tight_layout()
    fig.subplots_adjust(top=0.9) 
    plt.show()
    
def plot_model_data(description, func_eval, real_data, param_values):
    '''
    Plots a list of profiles in the same figure. Each profile corresponds
    to a simulation replica for the given instance.
    '''
    plt.rcParams["font.size"] = "10"
    N = len(param_values)
    D = description.shape[1]
    T = len(np.unique(description[:,0]))
    type_no = len(np.unique(description[:,1]))
    fig, axs = plt.subplots(type_no, figsize=(8, 12))

    for j in range(type_no):
        for i in range(N):
            p2 = axs[j].plot(range(T), func_eval[i,(j*T):(j*T + T)], color='grey')
        p1 = axs[j].plot(range(T), real_data[(j*T):(j*T + T)], 'ro' ,markersize = 5, color='red')
        if j == 0:
            axs[j].set_ylabel('COVID-19 Total Hospitalizations')
        elif j == 1:
            axs[j].set_ylabel('COVID-19 Hospital Admissions')
        elif j == 2:
            axs[j].set_ylabel('COVID-19 ICU Patients')
        axs[j].set_xlabel('Time (days)')
        axs[j].legend([p1[0], p2[0]], ['observations', 'computer model'])
    plt.show()

# (No Filter) Observe computer model outputs      
plot_model_data(description, func_eval_rnd, real_data, param_values_rnd)
    
# Filter out the data
par_out = param_values_rnd[np.logical_or.reduce((func_eval_rnd[:, 100] <= 200, func_eval_rnd[:, 20] >= 1000, func_eval_rnd[:, 100] >= 1000)),:]
par_in = param_values_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 200, func_eval_rnd[:, 20] < 1000, func_eval_rnd[:, 100] < 1000)), :]
func_eval_in = func_eval_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 200, func_eval_rnd[:, 20] < 1000, func_eval_rnd[:, 100] < 1000)), :]
par_in_test = param_values_test[np.logical_and.reduce((func_eval_test[:, 100] > 200, func_eval_test[:, 20] < 1000, func_eval_test[:, 100] < 1000)), :]
func_eval_in_test = func_eval_test[np.logical_and.reduce((func_eval_test[:, 100] > 200, func_eval_test[:, 20] < 1000, func_eval_test[:, 100] < 1000)), :]

#print(sps.describe(param_values))
# diff = func_eval_rnd - real_data
# diffsq = np.sum(diff**2, axis = 1)
# (Filter) Observe computer model outputs  
plot_model_data(description, func_eval_in, real_data, par_in)

x = np.hstack((np.reshape(np.tile(range(134), 3), (402, 1)),
              np.reshape(np.tile(np.array(('tothosp','totadmiss','icu')),134), (402, 1))))
x =  np.array(x, dtype='object')

# (No filter) Fit an emulator via 'PCGP'
emulator_1 = emulator(x = x,
                      theta = param_values_rnd,
                      f = func_eval_rnd.T,
                      method = 'PCGP') 

# (Filter) Fit an emulator via 'PCGP'
emulator_f_1 = emulator(x = x,
                        theta = par_in,
                        f = func_eval_in.T,
                        method = 'PCGP') 

# Define a class for prior of 10 parameters
class prior_covid:
    """ This defines the class instance of priors provided to the method. """
    #sps.uniform.logpdf(theta[:, 0], 3, 4.5)
    def lpdf(theta):
        return (sps.uniform.logpdf(theta[:, 0], 1.5, 2) +
                sps.uniform.logpdf(theta[:, 1], 3, 2) + 
                sps.uniform.logpdf(theta[:, 2], 3, 2) + 
                sps.uniform.logpdf(theta[:, 3], 1.25, 1.25) + 
                sps.uniform.logpdf(theta[:, 4], 10, 8) + 
                sps.uniform.logpdf(theta[:, 5], 14, 8) + 
                sps.uniform.logpdf(theta[:, 6], 16, 8) + 
                sps.uniform.logpdf(theta[:, 7], 10, 8) + 
                sps.uniform.logpdf(theta[:, 8], 9, 8) + 
                sps.uniform.logpdf(theta[:, 9], 8, 8)).reshape((len(theta), 1))


    def rnd(n):
        return np.vstack((sps.uniform.rvs(1.5, 2, size=n),
                          sps.uniform.rvs(3, 2, size=n),
                          sps.uniform.rvs(3, 2, size=n),
                          sps.uniform.rvs(1.25, 1.25, size=n),
                          sps.uniform.rvs(10, 8, size=n),
                          sps.uniform.rvs(14, 8, size=n),
                          sps.uniform.rvs(16, 8, size=n),
                          sps.uniform.rvs(10, 8, size=n),
                          sps.uniform.rvs(9, 8, size=n),
                          sps.uniform.rvs(8, 8, size=n))).T


##### ##### ##### ##### #####
# Run a classification model
cls_params = prior_covid.rnd(100000)
pred_1 = emulator_1.predict(x, cls_params)
pred_mean_1 = pred_1.mean()
y = np.zeros(len(pred_mean_1.T))
y[np.logical_and.reduce((pred_mean_1.T[:, 100] > 200, pred_mean_1.T[:, 20] < 1000, pred_mean_1.T[:, 100] < 1000))] = 1
 
# Create the test data
cls_params_test = prior_covid.rnd(5000)
pred_1_test = emulator_1.predict(x, cls_params_test)
pred_mean_1_test = pred_1_test.mean()
y_test = np.zeros(len(pred_mean_1_test.T))
y_test[np.logical_and.reduce((pred_mean_1_test.T[:, 100] > 200, pred_mean_1_test.T[:, 20] < 1000, pred_mean_1_test.T[:, 100] < 1000))] = 1

# Create a balanced data set

X_0 = cls_params[y == 0][0:20000]
y_0 = y[y == 0][0:20000]
X_1 = cls_params[y == 1]
y_1 = y[y == 1]
    
X = np.concatenate((X_0, X_1), axis=0)
y = np.concatenate((y_0, y_1), axis=0)

# Fit the classification model
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X, y)

#Training accuracy
print(model.score(X, y))
print(confusion_matrix(y, model.predict(X)))

#Test accuracy
print(model.score(cls_params_test, y_test))
print(confusion_matrix(y_test, model.predict(cls_params_test)))
##### ##### ##### ##### #####

    

    
obsvar = np.maximum(0.2*real_data, 5) 

cal_f = calibrator(emu = emulator_f_1,
                    y = real_data,
                    x = x,
                    thetaprior = prior_covid,
                    method = 'mlbayes',
                    yvar = obsvar, 
                    args = {'theta0': np.array([[2.5, 4, 4, 2, 14, 18, 20, 14, 13, 12]]), 
                            'numsamp' : 1000, 
                            'stepType' : 'normal', 
                            'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])})

plot_pred_interval(cal_f)
cal_f_theta = cal_f.theta.rnd(500) 
boxplot_param(cal_f_theta)

#breakpoint()
cal_f_ml = calibrator(emu = emulator_f_1,
                      y = real_data,
                      x = x,
                      thetaprior = prior_covid,
                      method = 'mlbayes',
                      yvar = obsvar, 
                      args = {'clf_method': model, 
                              'theta0': np.array([[2.5, 4, 4, 2, 14, 18, 20, 14, 13, 12]]), 
                              'numsamp' : 1000, 
                              'stepType' : 'normal', 
                              'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])})

plot_pred_interval(cal_f_ml)
cal_f_ml_theta = cal_f_ml.theta.rnd(500) 
boxplot_param(cal_f_ml_theta)
# breakpoint()
# cal_nf_ml = calibrator(emu = emulator_1,
#                         y = real_data,
#                         x = x,
#                         thetaprior = prior_covid,
#                         method = 'mlbayes',
#                         yvar = obsvar, 
#                         args = {'clf_method': model, 
#                                 'theta0': np.array([[2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]]), 
#                                 'numsamp' : 1000, 
#                                 'stepType' : 'normal', 
#                                 'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})

# plot_pred_interval(cal_nf_ml)
# cal_nf_ml_theta = cal_nf_ml.theta.rnd(1000) 
# boxplot_param(cal_nf_ml_theta)

# cal_nf = calibrator(emu = emulator_1,
#                         y = real_data,
#                         x = x,
#                         thetaprior = prior_covid,
#                         method = 'mlbayes',
#                         yvar = obsvar, 
#                         args = {'theta0': np.array([[2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]]), 
#                                 'numsamp' : 1000, 
#                                 'stepType' : 'normal', 
#                                 'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})

# plot_pred_interval(cal_nf)
# cal_nf_theta = cal_nf.theta.rnd(1000) 
# boxplot_param(cal_nf_theta)