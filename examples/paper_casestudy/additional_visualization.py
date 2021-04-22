import numpy as np
import matplotlib.pyplot as plt

def plot_classification_prob(prior_covid, samples, classification_model):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    n = 5000
    plt.xticks(fontsize=16)
    for i in range(2):
        for j in range(i+1, 3):
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)
    
    for id_i in range(3):
        for id_j in range(id_i+1,4):
            
            for idi in range(4):
                ml_sample = np.ones((n, 1)) * np.median(samples, axis=0)
    
            ml_sample[:, id_i] = prior_covid.rnd(n)[:, id_i] 
            ml_sample[:, id_j] = prior_covid.rnd(n)[:, id_j] 
               
            for i in range(0, 4):
                for j in range(i, 4):
                    ml_sample = np.concatenate([ml_sample, np.reshape(ml_sample[:, i] * ml_sample[:, j], (n, 1))], axis = 1)
                        
            yp = np.log(classification_model.predict_proba(ml_sample)[:, 1])
            yp -= np.max(yp)
            scatter = axs[id_j-1,id_i].scatter(ml_sample[:,id_i],
                                               ml_sample[:,id_j],
                                               c=yp,
                                               vmin=-20,
                                               vmax=0,
                                               cmap="Spectral")
            #legend1 = axs[id_j-1,id_i].legend(*scatter.legend_elements(num=5),
            #                    loc="lower right")
            #axs[id_j-1,id_i].add_artist(legend1)
            axs[id_j-1,id_i].tick_params(axis='x', labelsize= 16)
            axs[id_j-1,id_i].tick_params(axis='y', labelsize= 16)
    
    fig.colorbar(scatter, ax=axs[:,:])
    #labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
    labels = [r'$\sigma$', r'$\omega_A$', r'$\gamma_Y$', r'$\gamma_A$']
    axs[0, 0].set_ylabel(labels[1], fontsize=16)
    axs[1, 0].set_ylabel(labels[2], fontsize=16)
    axs[2, 0].set_ylabel(labels[3], fontsize=16)
    
    axs[2, 0].set_xlabel(labels[0], fontsize=16)
    axs[2, 1].set_xlabel(labels[1], fontsize=16)
    axs[2, 2].set_xlabel(labels[2], fontsize=16)
    #fig.tight_layout(pad=10.0)

    #plt.subplots_adjust(wspace = 0.5)
    plt.show()


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

def plot_loglikelihood(prior_covid, samples, obsvar, emulator_f_PCGPwM, real_data_tr, xtr, log_likelihood):
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    n = 5000
    
    for i in range(2):
        for j in range(i+1, 3):
            #print(i)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)
    
    for id_i in range(3):
        for id_j in range(id_i+1,4):
            
            for idi in range(4):
                ml_sample = np.ones((n, 1)) * np.median(samples, axis=0)
    
            ml_sample[:, id_i] = prior_covid.rnd(n)[:, id_i] 
            ml_sample[:, id_j] = prior_covid.rnd(n)[:, id_j] 
               
            yp = np.zeros((n))
            for i in range(n):
                yp[i] = log_likelihood(ml_sample[i,:], obsvar, emulator_f_PCGPwM, np.sqrt(real_data_tr), xtr)

            yp -= np.max(yp)
            scatter = axs[id_j-1,id_i].scatter(ml_sample[:,id_i], ml_sample[:,id_j], c=yp,
                                               vmin=-400,
                                               vmax=0,
                                               cmap="Spectral")
            #legend1 = axs[id_j-1,id_i].legend(*scatter.legend_elements(num=5),
            #                    loc="lower right")
            #axs[id_j-1,id_i].add_artist(legend1)
            axs[id_j-1,id_i].tick_params(axis='x', labelsize= 16)
            axs[id_j-1,id_i].tick_params(axis='y', labelsize= 16)
    
    fig.colorbar(scatter, ax=axs[:,:])
    labels = [r'$\sigma$', r'$\omega_A$', r'$\gamma_Y$', r'$\gamma_A$']
    axs[0, 0].set_ylabel(labels[1], fontsize=16)
    axs[1, 0].set_ylabel(labels[2], fontsize=16)
    axs[2, 0].set_ylabel(labels[3], fontsize=16)
    
    axs[2, 0].set_xlabel(labels[0], fontsize=16)
    axs[2, 1].set_xlabel(labels[1], fontsize=16)
    axs[2, 2].set_xlabel(labels[2], fontsize=16)
    
    #plt.subplots_adjust(wspace = 0.5)
    plt.show()



def plot_adjustedlikelihood(prior_covid, samples, obsvar, emulator_f_PCGPwM, real_data_tr, xtr, log_likelihood, classification_model):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    n = 5000
    
    for i in range(2):
        for j in range(i+1, 3):
            #print(i)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)
            
    for id_i in range(3):
        for id_j in range(id_i+1,4):
            
            for idi in range(4):
                ml_sample = np.ones((n, 1)) * np.median(samples, axis=0)
    
            ml_sample[:, id_i] = prior_covid.rnd(n)[:, id_i] 
            ml_sample[:, id_j] = prior_covid.rnd(n)[:, id_j] 
               
            yp = np.zeros((n))
            for i in range(n):
                yp[i] = log_likelihood(ml_sample[i,:], obsvar, emulator_f_PCGPwM, np.sqrt(real_data_tr), xtr)
            
            yp = np.max(yp)/yp
            
            for i in range(0, 4):
                for j in range(i, 4):
                    ml_sample = np.concatenate([ml_sample, np.reshape(ml_sample[:, i] * ml_sample[:, j], (len(ml_sample), 1))], axis = 1)
                        
            ypc = classification_model.predict_proba(ml_sample)[:, 1]             
            pp = yp*ypc

            scatter = axs[id_j-1,id_i].scatter(ml_sample[:,id_i], ml_sample[:,id_j], c=pp, cmap="Spectral")
            legend1 = axs[id_j-1,id_i].legend(*scatter.legend_elements(num=5),
                                loc="upper left", title="log-lik")
            axs[id_j-1,id_i].add_artist(legend1)
    
    labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
    axs[0, 0].set_ylabel(labels[1], fontsize=12)
    axs[1, 0].set_ylabel(labels[2], fontsize=12)
    axs[2, 0].set_ylabel(labels[3], fontsize=12)
    
    axs[2, 0].set_xlabel(labels[0], fontsize=12)
    axs[2, 1].set_xlabel(labels[1], fontsize=12)
    axs[2, 2].set_xlabel(labels[2], fontsize=12)
    plt.show()