import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
    
def boxplot_param(theta):
    plt.rcParams["font.size"] = "16"
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
    plt.rcParams["font.size"] = "10"
    fig, axs = plt.subplots(3, figsize=(8, 12))
    dim = int(len(x)/3)
    for j in range(3):
        upper = np.percentile(rndm_m[:, j*dim : (j + 1)*dim], 97.5, axis = 0)
        lower = np.percentile(rndm_m[:, j*dim : (j + 1)*dim], 2.5, axis = 0)
        median = np.percentile(rndm_m[:, j*dim : (j + 1)*dim], 50, axis = 0)
        p1 = axs[j].plot(median, color = 'black')
        axs[j].fill_between(range(0, dim), lower, upper, color = 'grey')
        p3 = axs[j].plot(range(0, dim), real_data[j*dim : (j + 1)*dim], 'ro' ,markersize = 5, color='red')
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
            axs[j].set_ylabel('COVID-19 ICU Patients')
        elif j == 2:
            axs[j].set_ylabel('COVID-19 Hospital Admissions')
        axs[j].set_xlabel('Time (days)')
        axs[j].legend([p1[0], p2[0]], ['observations', 'computer model'])
    plt.show()
    
def pair_scatter(params):

    df = pd.DataFrame(params, columns = ['Column_A','Column_B','Column_C','Column_D'])
    sns.set(font_scale=1.6)
    sns.set_style(style='white')
    g = sns.pairplot(df, corner = False, plot_kws=dict(marker="+", linewidth=1, color = 'grey'), diag_kws=dict(color = 'grey', bins=10))
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)   