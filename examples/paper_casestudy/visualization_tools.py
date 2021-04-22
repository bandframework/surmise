import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
  

def boxplot_compare(theta1, theta2):
    plt.rcParams["font.size"] = "12"
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))

    labels = [r'$\sigma$', r'$\omega_A$', r'$\gamma_Y$', r'$\gamma_A$']
    for i in range(4):
        axs[i].boxplot([theta1[:, i], theta2[:, i]], widths=(0.6, 0.6))
        axs[i].set_title(labels[i], fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05, top=0.95)
    plt.show()

def boxplot_param(theta):
    plt.rcParams["font.size"] = "8"
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    paraind = 0
    for i in range(2):
        for j in range(2):
            axs[i, j].boxplot(theta[:, paraind])
            paraind += 1

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05, top=0.95)
    plt.show()

def plot_pred_interval(cal, x, real_data):
    pr = cal.predict(x)
    rndm_m = pr.rnd(s = 100)
    plt.rcParams["font.size"] = "8"
    fig, axs = plt.subplots(3, figsize=(8, 12))
    dim = int(len(x)/3)
    for j in range(3):
        if j == 0:
            v = 'total_hosp'
        elif j == 1:
            v = 'icu_admission'
        else:
            v = 'daily_admission'
        
        ids = x[:, 1] == v
        y = real_data[ids]
        dim = len(y)
        
        upper = np.percentile(rndm_m[:, ids], 97.5, axis = 0)
        lower = np.percentile(rndm_m[:, ids], 2.5, axis = 0)
        median = np.percentile(rndm_m[:, ids], 50, axis = 0)
        p1 = axs[j].plot(median, color = 'black')
        axs[j].fill_between(range(0, dim), lower, upper, color = 'grey')
        p3 = axs[j].plot(range(0, dim), y, 'ro' ,markersize = 5, color='red')
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
 
def plot_pred_errors(cal, xtest, real_data_test):
    pr = cal.predict(xtest)
    rndm_m = pr.rnd(s = 100)
    plt.rcParams["font.size"] = "12"
    fig, axs = plt.subplots(3, figsize=(8, 12))

    for j in range(3):
        if j == 0:
            v = 'total_hosp'
        elif j == 1:
            v = 'icu_admission'
        else:
            v = 'daily_admission'
        
        ids = xtest[:, 1] == v
        y = real_data_test[ids]
        dim = len(y)
        
        y = y**2
        upper = np.percentile(rndm_m[:, ids], 97.5, axis = 0)
        lower = np.percentile(rndm_m[:, ids], 2.5, axis = 0)
        median = np.percentile(rndm_m[:, ids], 50, axis = 0)
        
        upper = upper**2
        lower = lower**2
        median = median**2
        
        p1 = axs[j].errorbar(range(0, dim), median, yerr = [median-lower, upper-median], ecolor="grey", color = 'black')
        #p1 = axs[j].plot(median, color = 'black')
        #axs[j].fill_between(range(0, dim), lower, upper, color = 'grey')
        p3 = axs[j].plot(range(0, dim), y, 'ro' ,markersize = 5, color='red')
        if j == 0:
            axs[j].set_ylabel('COVID-19 Total Hospitalizations')
        elif j == 1:
            axs[j].set_ylabel('COVID-19 ICU Patients')
        elif j == 2:
            axs[j].set_ylabel('COVID-19 Hospital Admissions')
        axs[j].set_xlabel('Time (days)')  
    
        axs[j].legend([p1[0], p3[0]], ['mean','observations'])
    fig.tight_layout()
    fig.subplots_adjust(top=0.9) 
    plt.show()

def plot_pred_errors_emcee(theta, emu, xtest, real_data_test, title):
    pr = emu.predict(theta=theta, x=xtest)
    rndm_m = pr.mean()
    plt.rcParams["font.size"] = "12"
    fig, axs = plt.subplots(3, figsize=(8, 12))

    for j in range(3):
        if j == 0:
            v = 'total_hosp'
            axs[j].set_title(title, fontsize=16)
        elif j == 1:
            v = 'icu_admission'
        else:
            v = 'daily_admission'
        
        ids = xtest[:, 1] == v
        y = real_data_test[ids]
        dim = len(y)
        
        y = y**2
        upper = np.percentile(rndm_m[ids, :], 97.5, axis = 1)
        lower = np.percentile(rndm_m[ids, :], 2.5, axis = 1)
        median = np.percentile(rndm_m[ids, :], 50, axis = 1) 

        upper = upper**2
        lower = lower**2
        median = median**2
        
        p1 = axs[j].errorbar(range(0, dim), median, yerr = [median-lower, upper-median], ecolor="grey", color = 'black')
        #p2 = axs[j].plot(range(0, dim), median, color = 'black')
        #p1 = axs[j].plot(median, color = 'black')
        #axs[j].fill_between(range(0, dim), lower, upper, color = 'grey')
        p3 = axs[j].plot(range(0, dim), y, 'ro' ,markersize = 5, color='red')
        if j == 0:
            axs[j].set_ylabel('COVID-19 Total Hospitalizations')
        elif j == 1:
            axs[j].set_ylabel('COVID-19 ICU Patients')
        elif j == 2:
            axs[j].set_ylabel('COVID-19 Hospital Admissions')
        axs[j].set_xlabel('Time (days)')  
        print(p1[0])
        axs[j].legend([p1[0], p3[0]], ['mean', 'observations'])
    
    fig.tight_layout()
    #fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.9) 
    plt.show()
    
def plot_model_data(description, func_eval, real_data, param_values):
    '''
    Plots a list of profiles in the same figure. Each profile corresponds
    to a simulation replica for the given instance.
    '''
    plt.rcParams["font.size"] = "12"
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
            axs[j].set_ylabel('COVID-19 ICU Patients')
        elif j == 2:
            axs[j].set_ylabel('COVID-19 Hospital Admissions')
        axs[j].set_xlabel('Time (days)')
        axs[j].legend([p1[0], p2[0]], ['observations', 'simulator output'])
    plt.show()
    
def pair_scatter(params):

    df = pd.DataFrame(params, columns = ['Column_A','Column_B','Column_C','Column_D'])
    sns.set(font_scale=1.6)
    sns.set_style(style='white')
    g = sns.pairplot(df, corner = False, plot_kws=dict(marker="+", linewidth=1, color = 'grey'), diag_kws=dict(color = 'grey', bins=10))
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
        
def plot_pred_interval_emce(emu, samples, x, real_data):
    mean_pred = np.zeros((len(samples), len(real_data)))
    for j in range(len(samples)):
        mean_pred[j, :] = emu.predict(x=x, theta=samples[j,:]).mean().reshape((len(real_data),))
        
    plt.rcParams["font.size"] = "8"
    fig, axs = plt.subplots(3, figsize=(8, 12))

    #real_data = real_data**2
    for j in range(3):
        if j == 0:
            v = 'total_hosp'
        elif j == 1:
            v = 'icu_admission'
        else:
            v = 'daily_admission'
        
        ids = x[:, 1] == v
        y = real_data[ids]
        dim = len(y)
        
        upper = np.percentile(mean_pred[:, ids], 97.5, axis = 0)
        lower = np.percentile(mean_pred[:, ids], 2.5, axis = 0)
        median = np.percentile(mean_pred[:, ids], 50, axis = 0)
        
        #upper = upper**2
        #lower = lower**2
        #median = median**2
        p1 = axs[j].plot(median, color = 'black')
        axs[j].fill_between(range(0, dim), lower, upper, color = 'grey')
        p3 = axs[j].plot(range(0, dim), real_data[ids], 'ro' ,markersize = 5, color='red')
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
    