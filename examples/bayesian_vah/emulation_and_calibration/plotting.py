#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 21:20:08 2022

@author: ozgesurer
"""
from matplotlib.transforms import Transform
import numpy as np
from sklearn import metrics
#import uncertainty_toolbox as uct
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import os
import pandas as pd

# 8 bins
ALICE_cent_bins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])

obs_cent_list = {
'Pb-Pb-2760': {
    'dNch_deta' : ALICE_cent_bins,
    'dET_deta' : np.array([[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10],
                           [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20],
                           [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30],
                           [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40],
                           [40, 45], [45, 50], [50, 55], [55, 60],
                           [60, 65], [65, 70]]), # 22 bins
    'dN_dy_pion'   : ALICE_cent_bins,
    'dN_dy_kaon'   : ALICE_cent_bins,
    'dN_dy_proton' : ALICE_cent_bins,
    'dN_dy_Lambda' : np.array([[0,5],[5,10],[10,20],[20,40],[40,60]]), # 5 bins
    'dN_dy_Omega'  : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'dN_dy_Xi'     : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'mean_pT_pion'   : ALICE_cent_bins,
    'mean_pT_kaon'   : ALICE_cent_bins,
    'mean_pT_proton' : ALICE_cent_bins,
    'pT_fluct' : np.array([[0,5],[5,10],[10,15],[15,20], [20,25],[25,30],[30,35],[35,40], [40,45],[45,50],[50,55],[55,60]]), #12 bins
    'v22' : ALICE_cent_bins,
    'v32' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    'v42' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    }
}

obs_groups = {'yields' : ['dNch_deta', 'dET_deta', 'dN_dy_pion', 'dN_dy_kaon', 'dN_dy_proton'],
              'mean_pT' : ['mean_pT_pion', 'mean_pT_kaon','mean_pT_proton', ],
              'fluct' : ['pT_fluct'],
              'flows' : ['v22', 'v32', 'v42']}

obs_group_labels = {'yields' : r'$dN_\mathrm{id}/dy_p$, $dN_\mathrm{ch}/d\eta$, $dE_T/d\eta$ [GeV]',
                    'mean_pT' : r'$ \langle p_T \rangle_\mathrm{id}$' + ' [GeV]',
                    'fluct' : r'$\delta p_{T,\mathrm{ch}} / \langle p_T \rangle_\mathrm{ch}$',
                    'flows' : r'$v^{(\mathrm{ch})}_k\{2\} $'}

colors = ['b', 'g', 'r', 'c', 'm', 'tan', 'gray']

obs_tex_labels = {'dNch_deta' : r'$dN_\mathrm{ch}/d\eta$',
                  'dN_dy_pion' : r'$dN_{\pi}/dy_p$',
                  'dN_dy_kaon' : r'$dN_{K}/dy_p$',
                  'dN_dy_proton' : r'$dN_{p}/dy_p$',
                  'dET_deta' : r'$dE_{T}/d\eta$',

                  'mean_pT_proton' : r'$\langle p_T \rangle_p$',
                  'mean_pT_kaon' : r'$\langle p_T \rangle_K$',
                  'mean_pT_pion' : r'$\langle p_T \rangle_\pi$',

                  'pT_fluct' : r'$\delta p_{T,\mathrm{ch}} / \langle p_T \rangle_\mathrm{ch}$',
                  'v22' : r'$v^{(\mathrm{ch})}_2\{2\}$',
                  'v32' : r'$v^{(\mathrm{ch})}_3\{2\}$',
                  'v42' : r'$v^{(\mathrm{ch})}_4\{2\}$'}


# Map the design names to proper form
# model_param_dsgn = ['$N$[$2.76$TeV]',
#  '$p$',
#  '$w$ [fm]',
#  '$d_{\\mathrm{min}}$ [fm]',
#  '$\\sigma_k$',
#  '$T_{\\mathrm{sw}}$ [GeV]',
#  '$T_{\\eta,\\mathrm{kink}}$ [GeV]',
#  '$(\\eta/s)_{\\mathrm{kink}}$',
#  '$a_{\\eta,\\mathrm{low}}$ [GeV${}^{-1}$]',
#  '$a_{\\eta,\\mathrm{high}}$ [GeV${}^{-1}$]',
#  '$(\\zeta/s)_{\\max}$',
#  '$T_{\\zeta,c}$ [GeV]',
#  '$w_{\\zeta}$ [GeV]',
#  '$\\lambda_{\\zeta}$',
#  '$R$']

# Map the design names to proper form
model_param_dsgn = ['$N$',
 '$p$',
 '$w$',
 '$d_{\\mathrm{min}}$',
 '$\\sigma_k$',
 '$T_{\\mathrm{sw}}$',
 '$T_{\\eta,\\mathrm{kink}}$',
 '$(\\eta/s)_{\\mathrm{kink}}$',
 '$a_{\\eta,\\mathrm{low}}$',
 '$a_{\\eta,\\mathrm{high}}$',
 '$(\\zeta/s)_{\\max}$',
 '$T_{\\zeta,c}$',
 '$w_{\\zeta}$',
 '$\\lambda_{\\zeta}$',
 '$R$']

index={}
st_index=0
for obs_group in  obs_groups.keys():
    for obs in obs_groups[obs_group]:
        n_centrality = len(obs_cent_list['Pb-Pb-2760'][obs])
        index[obs]=[st_index,st_index+n_centrality]
        st_index = st_index + n_centrality

def plot_UQ(f, fhat, sigmahat, method='PCGP', drop=[None]):
    
    index={}
    st_index=0
    for obs_group in  obs_groups.keys():
        if obs_group in drop:
            continue;
        for obs in obs_groups[obs_group]:
            n_centrality = len(obs_cent_list['Pb-Pb-2760'][obs])
            index[obs]=[st_index,st_index+n_centrality]
            st_index = st_index + n_centrality
    
    
    r = metrics.r2_score(fhat.flatten(), f.flatten())
    print('R2 test(sklearn) = ',r)
    
    
    sns.set_context('paper', font_scale=0.8)
    for obs in index.keys():
        st = index[obs][0]
        ed = index[obs][1]
        nrw = int(np.ceil((ed-st)/4))
        fig, axs = plt.subplots(nrows=nrw, ncols=4, figsize=(10, nrw*4), sharex=True, sharey=True)
        for iii,ax in enumerate(axs.flatten()):
            if iii>=ed-st:
                continue;
            ii = st + iii
            mse = sklearn.metrics.mean_squared_error(f[:, ii], fhat[:, ii])
            r = sklearn.metrics.r2_score(f[:, ii], fhat[:, ii])
            uct.plot_calibration(fhat[:, ii], sigmahat[:, ii], f[:, ii], ax=ax)
            cen_st = obs_cent_list['Pb-Pb-2760'][obs][iii]
            ax.set_title(f'{cen_st}  R2: {r:.2f}')
        fig.suptitle(obs_tex_labels[obs])
        plt.tight_layout()
        os.makedirs(f'{method}', exist_ok=True)
        plt.savefig(f'{method}/{obs}.png', dpi=200)
        #plt.close('all')
        #plt.show()
        
        
    
    sns.set_context('paper', font_scale=0.8)
    for obs in index.keys():
        st = index[obs][0]
        ed = index[obs][1]
        nrw = int(np.ceil((ed-st)/4))
        fig, axs = plt.subplots(nrows=nrw, ncols=4, figsize=(10, nrw*4), sharex=False, sharey=False)
        for iii,ax in enumerate(axs.flatten()):
            
            if iii>=ed-st:
                continue;
            ii=st+iii
            mse = sklearn.metrics.mean_squared_error(f[:, ii], fhat[:, ii])
            r = sklearn.metrics.r2_score(f[:, ii], fhat[:, ii])
            ax.errorbar(x=f[:,ii], y=fhat[:,ii], yerr=sigmahat[:, ii], fmt='x')
            min_value = min([ax.get_xlim()[0], ax.get_ylim()[0]])
            max_value = min([ax.get_xlim()[1], ax.get_ylim()[1]])
            ax.plot([min_value, max_value], [min_value, max_value])
            ax.set_xlabel('Simulation')
            ax.set_ylabel('Emulation')
            cen_st = obs_cent_list['Pb-Pb-2760'][obs][iii]
            ax.set_title(f'{cen_st}  R2: {r:.2f}')
            fig.suptitle(obs_tex_labels[obs])
            plt.tight_layout()
            os.makedirs(f'{method}/emu_vs_sim/', exist_ok=True)
            plt.savefig(f'{method}/emu_vs_sim/{obs}.png', dpi=200)
            #plt.close('all')
    return None


def plot_R2(fhat, f, method, drop=None):
    index={}
    st_index=0
    for obs_group in  obs_groups.keys():
        if obs_group in drop:
            continue;
        for obs in obs_groups[obs_group]:
            n_centrality = len(obs_cent_list['Pb-Pb-2760'][obs])
            index[obs]=[st_index,st_index+n_centrality]
            st_index = st_index + n_centrality

    sns.set_context('poster',font_scale=0.5)
    rsq = []
    for i in range(fhat.shape[0]):
        sse = np.sum((fhat[i, :] - f[i, :])**2)
        sst = np.sum((f[i, :] - np.mean(f[i, :]))**2)
        rsq.append(1 - sse/sst)
    fig, ax = plt.subplots()
    #ax.scatter(np.arange(fhat.shape[0]), rsq)
    for k in index.keys():
        low = index[k][0]
        high = index[k][1]
        ax.scatter(np.arange(low,high),rsq[low:high], label = k, s=3)
    ax.set_xlabel('Observables')
    ax.set_ylabel(r'Test $R^2$')

    ax.set_xticks([int(np.ceil((index[k][0]+index[k][1])/2)) for k in index.keys()])
    ax.set_xticklabels([obs_tex_labels[k] for k in index.keys()], rotation=45)
    plt.tight_layout()
    
    os.makedirs(f'{method}', exist_ok=True)
    plt.savefig(f'{method}/R2.png', dpi=200)
    plt.show()
    np.save(f'{method}/R2_values', rsq)
    return None
      
def plot_hist(theta_prior, theta_post, method):
    fig, axs = plt.subplots(5, 3, figsize=(16, 16))
    theta_prior.hist(ax=axs)
    theta_post.hist(ax=axs, bins=25)
    plt.savefig(f'{method}/hist.png', dpi=200)

def plot_density(theta_prior, theta_post, thetanames, method):
    dfpost = pd.DataFrame(theta_post, columns = thetanames)
    dfprior = pd.DataFrame(theta_prior, columns = thetanames)
    df = pd.concat([dfprior, dfpost])
    pr = ['prior' for i in range(1000)]
    ps = ['posterior' for i in range(1000)]
    pr.extend(ps)
    df['distribution'] = pr
    sns.set_context('poster', font_scale=1)
    sns.set(style="white")
    g = sns.PairGrid(df, corner=True, diag_sharey=False, hue='distribution')    
    g.map_diag(sns.kdeplot, shade=True)
    g.map_lower(sns.kdeplot, fill=True)
    plt.savefig(f'{method}/density.png', dpi=200)
    return None

def plot_corner_viscosity(posterior_df,prior_df, method_name, n_samples=1000, prune=1, MAP=None, closure=None):

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    #map_parameters = rslt.x
    sns.set_palette('bright')
    observables_to_plot=[6, 7 , 8, 9, 10, 11, 12, 13 ]
    posterior_df = posterior_df.copy(deep=True)
    prior_df = prior_df.copy(deep=True)
    posterior_df['distribution'] = 'posterior'
    prior_df['distribution'] = 'prior'
    df = pd.concat([prior_df, posterior_df], ignore_index=True)
    obs = observables_to_plot + [15]
    df = df.iloc[0:2*n_samples:prune,obs] 
    g = sns.PairGrid(df, corner=True, diag_sharey=False, hue='distribution', hue_kws={'alpha':0.5},
                    palette={'prior':sns.color_palette()[4],'posterior':sns.color_palette()[5]})
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.kdeplot, linewidth=2, shade=True , fill=True)
    for n,i in enumerate(observables_to_plot):
        ax=g.axes[n][n]
        if MAP is not None:
            map_parameters=MAP.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9], label= 'MAP')
            ax.text(0,0.9,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)
        if closure is not None:
            map_parameters=closure.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[3], label= 'Truth')
            ax.text(0,0.7,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)        
    #ax.axvline(x=truth[i], ls='--', c=sns.color_palette()[3], label = 'Truth')
    #ax.text(0,0.8,s= f'{truth[i]:.3f}', transform=ax.transAxes)
        if n==4:
            ax.legend(loc=0,fontsize='xx-small')    
    plt.tight_layout()
    plt.savefig(f'{method_name}/Viscosity.png', dpi=200)
    plt.show()
    return None


def plot_corner_no_viscosity(posterior_df,prior_df,  method_name, n_samples = 1000, prune=1, MAP=None, closure=None, transform=False):

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    #map_parameters=map_values_saved.flatten()
    #n_samples_prior = 20000
    #prune = 1
    sns.set_palette('bright')
    if transform==False:
        observables_to_plot=[0, 1, 2 ,3 , 4, 5, 14]
        obs = observables_to_plot + [15]
    else:
        observables_to_plot=[0, 1, 2 ,3 , 4, 5, 6]
        obs = observables_to_plot + [27]
    posterior_df = posterior_df.copy(deep=True)
    prior_df = prior_df.copy(deep=True)
    posterior_df['distribution'] = 'posterior'
    prior_df['distribution'] = 'prior'
    df = pd.concat([prior_df, posterior_df], ignore_index=True)
    
    df = df.iloc[0:2*n_samples:prune,obs] 
    g = sns.PairGrid(df, corner=True, diag_sharey=False, hue='distribution', hue_kws={'alpha':0.5},
                    palette={'prior':sns.color_palette()[4],'posterior':sns.color_palette()[5]})
    g.map_lower(sns.kdeplot, fill=True)
    #g.map_upper(sns.kdeplot, shade=True, color=sns.color_palette()[0])
    g.map_diag(sns.kdeplot, linewidth=2, shade=True)
    for n,i in enumerate(observables_to_plot):
        ax=g.axes[n][n]
        if MAP is not None:
            map_parameters = MAP.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9], label='MAP')
            ax.text(0.0,1,s= f'{map_parameters[i]:.3f}',fontdict={'color':sns.color_palette()[9]}, transform=ax.transAxes)
        if closure is not None:
            map_parameters=closure.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[3], label= 'Truth')
            ax.text(0,0.7,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)    
    #ax.axvline(x=truth[i], ls='--', c=sns.color_palette()[3], label = 'Truth')
    #ax.text(0.6,1,s= f'{truth[i]:.3f}',fontdict={'color':sns.color_palette()[3]}, transform=ax.transAxes)
        if n==0:
            ax.legend(loc=1,fontsize='xx-small')
    plt.tight_layout()
    if transform==False:
        plt.savefig(f'{method_name}/WithoutViscosity.png', dpi=200)
    else:
        plt.savefig(f'{method_name}/WithoutViscosity_transform.png', dpi=200)
    plt.show()
    return None
   
def plot_corner_no_bulk_viscosity(posterior_df,prior_df,  method_name, n_samples = 1000, prune=1, MAP=None, closure=None, transform=False):

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    #map_parameters=map_values_saved.flatten()
    #n_samples_prior = 20000
    #prune = 1
    sns.set_palette('bright')
    if transform==False:
        observables_to_plot=[0, 1, 2 ,3 , 4, 5, 6, 7 , 8, 9,14]
        obs = observables_to_plot + [15]
    else:
        observables_to_plot=[0, 1, 2 ,3 , 4, 5, 6]
        obs = observables_to_plot + [27]
    posterior_df = posterior_df.copy(deep=True)
    prior_df = prior_df.copy(deep=True)
    posterior_df['distribution'] = 'posterior'
    prior_df['distribution'] = 'prior'
    df = pd.concat([prior_df, posterior_df], ignore_index=True)
    
    df = df.iloc[0:2*n_samples:prune,obs] 
    g = sns.PairGrid(df, corner=True, diag_sharey=False, hue='distribution', hue_kws={'alpha':0.5},
                    palette={'prior':sns.color_palette()[4],'posterior':sns.color_palette()[5]})
    g.map_lower(sns.kdeplot, fill=True)
    #g.map_upper(sns.kdeplot, shade=True, color=sns.color_palette()[0])
    g.map_diag(sns.kdeplot, linewidth=2, shade=True)
    for n,i in enumerate(observables_to_plot):
        ax=g.axes[n][n]
        if MAP is not None:
            map_parameters = MAP.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9], label='MAP')
            ax.text(0.0,1,s= f'{map_parameters[i]:.3f}',fontdict={'color':sns.color_palette()[9]}, transform=ax.transAxes)
        if closure is not None:
            map_parameters=closure.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[3], label= 'Truth')
            ax.text(0,0.7,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)    
    #ax.axvline(x=truth[i], ls='--', c=sns.color_palette()[3], label = 'Truth')
    #ax.text(0.6,1,s= f'{truth[i]:.3f}',fontdict={'color':sns.color_palette()[3]}, transform=ax.transAxes)
        if n==0:
            ax.legend(loc=1,fontsize='xx-small')
    plt.tight_layout()
    if transform==False:
        plt.savefig(f'{method_name}/WithoutBulkViscosity.png', dpi=200)
    else:
        plt.savefig(f'{method_name}/WithoutBulkViscosity_transform.png', dpi=200)
    plt.show()
    return None


def plot_corner_all(posterior_df, prior_df, method_name, n_samples = 1000, prune=1, MAP=None, closure=None, transform=False):
    sns.set_context("notebook", font_scale=2.0)
    plt.rcParams["axes.labelsize"] = 30
    sns.set_style("ticks")
    #map_parameters=map_values_saved.flatten()
    #n_samples_prior = 20000
    #prune = 1
    #map_parameters = rslt.x
    sns.set_palette('bright')
    if transform==False:
        observables_to_plot=[i for i in range(0,15)]
        obs = observables_to_plot + [15]
    else:
        observables_to_plot=[i for i in range(0,27)]
        obs = observables_to_plot + [27]   
    
    posterior_df = posterior_df.copy(deep=True)
    prior_df = prior_df.copy(deep=True)
    posterior_df['distribution'] = 'posterior'
    prior_df['distribution'] = 'prior'
    df = pd.concat([prior_df, posterior_df], ignore_index=True)
    df = df.iloc[0:2*n_samples:prune,obs] 
    g = sns.PairGrid(df, corner=True, diag_sharey=False, hue='distribution', hue_kws={'alpha':0.5},
                    palette={'prior':sns.color_palette()[4],'posterior':sns.color_palette()[5]})
    g.map_lower(sns.kdeplot, fill=True)
    g.map_lower(sns.kdeplot, fill=True)
    #g.map_upper(sns.kdeplot, shade=True, color=sns.color_palette()[0])
    g.map_diag(sns.kdeplot, linewidth=2, shade=True, color=sns.color_palette()[4])
    for n,i in enumerate(observables_to_plot):
        ax=g.axes[n][n]
        if MAP is not None:
            map_parameters = MAP.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9], label='MAP')
            ax.text(0.0,1,s= f'{map_parameters[i]:.3f}',fontdict={'color':sns.color_palette()[9]}, transform=ax.transAxes)
        if closure is not None:
            map_parameters=closure.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[3], label= 'Truth')
            ax.text(0,0.7,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)    
    #ax.axvline(x=truth[i], ls='--', c=sns.color_palette()[3], label = 'Truth')
    #ax.text(0.6,1,s= f'{truth[i]:.3f}',fontdict={'color':sns.color_palette()[3]}, transform=ax.transAxes)
        if n==0:
            ax.legend(loc=1,fontsize='xx-small')
    plt.tight_layout()
    if transform==False:
        plt.savefig(f'{method_name}/all.png', dpi=200)
    else:
        plt.savefig(f'{method_name}/all_transform.png', dpi=200)
    plt.show()
    return None

def zeta_over_s(T, zmax, T0, width, asym):
    DeltaT = T - T0
    sign = 1 if DeltaT>0 else -1
    x = DeltaT/(width*(1.+asym*sign))
    return zmax/(1.+x**2)
zeta_over_s = np.vectorize(zeta_over_s)

def eta_over_s(T, T_k, alow, ahigh, etas_k):
    if T < T_k:
        y = etas_k + alow*(T-T_k)
    else:
        y = etas_k + ahigh*(T-T_k)
    if y > 0:
        return y
    else:
        return 0.
eta_over_s = np.vectorize(eta_over_s)


def plot_shear(posterior_df, method_name, prior, n_samples = 1000, prune=1, MAP=None, closure=None, ax= None, legend=False):
    sns.set_context('paper', font_scale=2)
    Tt = np.linspace(0.15, 0.35, 100)
    if ax==None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6), sharex=False, sharey=False, constrained_layout=True)
        #fig.suptitle("Specific shear viscosity posterior", wrap=True)
        fig.suptitle("(a)", wrap=True)

    else:
        axes=ax




    prior_etas = []
    design_min, design_max = prior[:,0], prior[:,1]
    for row in np.random.uniform(design_min, design_max,(10000,15))[:,[6,7,8,9]]:
        [T_k, etas_k, alow, ahigh] = row
        prior=[]
        for T in Tt:
            prior.append(eta_over_s(T,T_k,alow,ahigh,etas_k))
        prior_etas.append(prior)
    prior_percentile = np.percentile(prior_etas,[0,5,20,80,95,100], axis=0)
    np.save(f'{method_name}/shear_prior', prior_percentile)
    per0_pr,per5_pr,per20_pr,per80_pr,per95_pr,per100_pr=prior_percentile

    posterior_etas = []
    
    for row in posterior_df.iloc[0:n_samples:prune,[6,7,8,9]].values:
        [T_k, etas_k, alow, ahigh] = row
        posterior=[]
        for T in Tt:
            posterior.append(eta_over_s(T,T_k,alow,ahigh,etas_k))
        posterior_etas.append(posterior)
    pos_percentile = np.percentile(posterior_etas,[0,5,20,80,95,100], axis=0)
    np.save(f'{method_name}/shear_posterior', pos_percentile)
    per0,per5,per20,per80,per95,per100=pos_percentile
    
            
    axes.fill_between(Tt, per5_pr,per95_pr,color=sns.color_palette()[7], alpha=0.3, label='90% Prior')
    axes.fill_between(Tt,per5,per95,color=sns.color_palette()[9], alpha=0.2, label='90% C.I.')
    axes.fill_between(Tt,per20,per80, color=sns.color_palette()[9], alpha=0.3, label='60% C.I.')

    # Map, True temperature dependece of the viscosity
    if closure is not None:
        values = closure.flatten()
        print(values)
        [T_k, etas_k, alow, ahigh] = values[[6,7,8,9]]
        true_shear = eta_over_s(Tt, T_k, alow, ahigh, etas_k)
        axes.plot(Tt, true_shear, color = 'black', label = 'Truth', linewidth=2, linestyle='--')
    if MAP is not None:
        values = MAP.flatten()
        [T_k, etas_k, alow, ahigh] = values[[6,7,8,9]]
        true_shear = eta_over_s(Tt, T_k, alow, ahigh, etas_k)
        axes.plot(Tt, true_shear, color = 'g', label = 'MAP', linewidth=5)
    axes.legend(loc='upper left')
    axes.set_ylim(0,0.5)
    axes.set_xlabel('T [GeV]')
    axes.set_ylabel('$\eta/s$')
    
    plt.tight_layout()
    if method_name!=None:
        plt.savefig(f'{method_name}/shear.png', dpi=200)
        plt.show()
    return None

def plot_bulk(posterior_df, method_name, prior, n_samples = 1000, prune=1, MAP=None, closure=None, ax=None, legend=False):
    sns.set_context('paper', font_scale=2)
    Tt = np.linspace(0.15, 0.35, 100)
    if ax==None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6),sharex=False, sharey=False, constrained_layout=True)
        #fig.suptitle("Specefic bulk viscosity posterior", wrap=True)
        fig.suptitle("(b)", wrap=True)
    else:
        axes=ax

    # True temperature dependece of the viscosity

    #[zmax, T0, width, asym] = truth[[11,12,13,14]]
    #true_bulk = zeta_over_s(Tt, zmax, T0, width, asym)


    prior_zetas = []
    design_min, design_max = prior[:,0], prior[:,1]
    for row in np.random.uniform(design_min, design_max,(10000,15))[:,[10,11,12,13]]:
        [zmax, T0, width, asym] = row   
        prior=[]
        for T in Tt:
            prior.append(zeta_over_s(T,zmax, T0, width, asym))
        prior_zetas.append(prior)
    prior_percentile=np.percentile(prior_zetas,[0,5,20,80,95,100], axis=0)
    np.save(f'{method_name}/bulk_prior', prior_percentile)
    per0_pr,per5_pr,per20_pr,per80_pr,per95_pr,per100_pr=prior_percentile

    posterior_zetas = []
        
    for row in posterior_df.iloc[0:n_samples:prune,[10,11,12,13]].values:
        [zmax, T0, width, asym] = row   
        posterior=[]
        for T in Tt:
            posterior.append(zeta_over_s(T,zmax, T0, width, asym))
        posterior_zetas.append(posterior)
    posterior_percentile=np.percentile(posterior_zetas,[0,5,20,80,95,100], axis=0)        
    np.save(f'{method_name}/bulk_posterior', posterior_percentile)
    per0,per5,per20,per80,per95,per100=posterior_percentile
    axes.fill_between(Tt, per5_pr,per95_pr,color=sns.color_palette()[7], alpha=0.3, label='90% Prior')
    axes.fill_between(Tt,per5,per95,color=sns.color_palette()[4], alpha=0.2, label='90% C.I.')
    axes.fill_between(Tt,per20,per80, color=sns.color_palette()[4], alpha=0.3, label='60% C.I.')

    if closure is not None:
        values = closure.flatten()
        [zmax, T0, width, asym] = values[[10,11,12,13]]
        true_bulk = zeta_over_s(Tt, zmax, T0, width, asym)
        axes.plot(Tt, true_bulk,  color = 'black', label = 'Truth', linewidth=2, linestyle='--')
    if MAP is not None:
        values = MAP.flatten()
        [zmax, T0, width, asym] = values[[10,11,12,13]]
        true_bulk = zeta_over_s(Tt, zmax, T0, width, asym)
        axes.plot(Tt, true_bulk, color = 'g', label = 'MAP', linewidth=5)
    #axes.plot(Tt, true_bulk, color = 'r', label = 'Truth', linewidth=5)

    #pos=np.array(prior_zetas).T
    #axes.violinplot(pos[1::10,:].T, positions=Tt[1::10],widths=0.03)
    
    axes.legend(loc='upper right')
    axes.set_ylim(0,0.25)
    #else:
        #axes.legend(loc='upper right')
    axes.set_xlabel('T [GeV]')
    axes.set_ylabel('$\zeta/s$')
    plt.tight_layout()
    if method_name!=None:
        plt.savefig(f'{method_name}/bulk.png', dpi=200)
        plt.show()
    return None

def plot_shear_tf(posterior_df, method_name, prior_df, n_samples = 1000, prune=1, MAP=None, closure=None):

    Tt = np.linspace(0.135, 0.4, 10)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6), sharex=False, sharey=False, constrained_layout=True)
    fig.suptitle("Specific shear viscosity posterior", wrap=True)

    prior_etas = prior_df.values[:,7:17]
    per0_pr,per5_pr,per20_pr,per80_pr,per95_pr,per100_pr=np.percentile(prior_etas,[0,5,20,80,95,100], axis=0)

    posterior_etas = posterior_df.values[:,17:27]
    per0,per5,per20,per80,per95,per100=np.percentile(posterior_etas,[0,5,20,80,95,100], axis=0)
    axes.fill_between(Tt, per5_pr,per95_pr,color=sns.color_palette()[6], alpha=0.1, label='90% Prior')
    axes.fill_between(Tt,per5,per95,color=sns.color_palette()[9], alpha=0.2, label='90% C.I.')
    axes.fill_between(Tt,per20,per80, color=sns.color_palette()[9], alpha=0.3, label='60% C.I.')

    # Map, True temperature dependece of the viscosity
    if closure is not None:
        values = closure.reshape(1,-1)
        print(values)
        true_shear = values[:,7:17].flatten()
        axes.plot(Tt, true_shear, color = 'r', label = 'Truth', linewidth=5)
    if MAP is not None:
        values = transform_design(MAP.reshape(1,-1))
        true_shear = values[:,7:17].flatten()
        axes.plot(Tt, true_shear, color = 'g', label = 'MAP', linewidth=5)

    axes.legend(loc='upper left')
#axes.set_ylim(0,1.2)
    axes.set_xlabel('T [GeV]')
    axes.set_ylabel('$\eta/s$')
    plt.tight_layout()
    plt.savefig(f'{method_name}/shear_transform.png', dpi=200)
    plt.show()
    return None

def plot_bulk_tf(posterior_df, method_name, prior_df, n_samples = 1000, prune=1, MAP=None, closure=None):
    
    Tt = np.linspace(0.135, 0.4, 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6),
                            sharex=False, sharey=False, constrained_layout=True)
    fig.suptitle("Specefic bulk viscosity posterior", wrap=True)

    # True temperature dependece of the viscosity

    #[zmax, T0, width, asym] = truth[[11,12,13,14]]
    #true_bulk = zeta_over_s(Tt, zmax, T0, width, asym)


    prior_zetas = prior_df.values[:,17:27]
    per0_pr,per5_pr,per20_pr,per80_pr,per95_pr,per100_pr=np.percentile(prior_zetas,[0,5,20,80,95,100], axis=0)

    posterior_zetas = posterior_df.values[:,17:27]

    per0,per5,per20,per80,per95,per100=np.percentile(posterior_zetas,[0,5,20,80,95,100], axis=0)
    axes.fill_between(Tt, per0_pr,per100_pr,color=sns.color_palette()[7], alpha=0.1, label='Prior')
    axes.fill_between(Tt,per5,per95,color=sns.color_palette()[4], alpha=0.2, label='90% C.I.')
    axes.fill_between(Tt,per20,per80, color=sns.color_palette()[4], alpha=0.3, label='60% C.I.')

    if closure is not None:
        values = closure.reshape(1,-1)
        true_bulk = values[:,17:27].flatten()
        axes.plot(Tt, true_bulk, color = 'r', label = 'Truth', linewidth=5)
    if MAP is not None:
        values = transform_design(MAP.reshape(1,-1))
        true_bulk = values[:,17:27].flatten()
        axes.plot(Tt, true_bulk, color = 'g', label = 'MAP', linewidth=5)
    #axes.plot(Tt, true_bulk, color = 'r', label = 'Truth', linewidth=5)

    #pos=np.array(prior_zetas).T
    #axes.violinplot(pos[1::10,:].T, positions=Tt[1::10],widths=0.03)

    axes.legend(loc='upper right')
    #axes.set_ylim(0,1.2)
    axes.set_xlabel('T [GeV]')
    axes.set_ylabel('$\zeta/s$')
    plt.tight_layout()
    plt.savefig(f'{method_name}/bulk_transform.png', dpi=200)
    plt.show()
    return None

def transform_design(X):
    #pop out the viscous parameters
    indices = [0, 1, 2, 3, 4, 5 , 14]
    new_design_X = X[:, indices]

    #now append the values of eta/s and zeta/s at various temperatures
    num_T = 10
    Temperature_grid = np.linspace(0.135, 0.4, num_T)
    eta_vals = []
    zeta_vals = []
    for pt, T in enumerate(Temperature_grid):
        eta_vals.append( eta_over_s(T, X[:, 6], X[:, 8], X[:, 9], X[:, 7]) )
    for pt, T in enumerate(Temperature_grid):
        zeta_vals.append( zeta_over_s(T, X[:, 10], X[:, 11], X[:, 12], X[:, 13]) )

    eta_vals = np.array(eta_vals).T
    zeta_vals = np.array(zeta_vals).T

    new_design_X = np.concatenate( (new_design_X, eta_vals), axis=1)
    new_design_X = np.concatenate( (new_design_X, zeta_vals), axis=1)
    return new_design_X
