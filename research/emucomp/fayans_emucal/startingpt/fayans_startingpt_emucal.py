import scipy.io as spio
import numpy as np
import scipy.stats as sps
from surmise.emulation import emulator
from surmise.calibration import calibrator

class prior_fayans:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        return sps.beta.logpdf(theta, 2, 2).sum(1).reshape((theta.shape[0], 1))

    def rnd(n):
        return sps.beta.rvs(2, 2, size=(n, 13))

import os
os.chdir(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\startingpt')

mat = spio.loadmat(r'starting_points_test_info.mat')
bigmap = np.loadtxt(r'errmap.txt', delimiter=',', dtype=int)
inputs = np.loadtxt(r'inputdata.csv', delimiter=',', dtype=object)

fvals = mat['Fhist']
errvals = mat['Errorhist']
thetavals = mat['X0mat'].T
obsvals = mat['fvals'].T
toterr = errvals @ bigmap
errvalssimple = toterr > 0.5

fvals[errvalssimple] = np.nan

n = 500
thetanorm = np.linalg.norm(thetavals-0.5, ord=1, axis=1)
thetatopinds = np.argpartition(thetanorm, -n)[-n:]

# subset training data
ftrain = fvals[thetatopinds]
thetatrain = thetavals[thetatopinds]
testinds = np.setdiff1d(np.arange(1000), thetatopinds)
ftest = fvals[testinds]
thetatest = thetavals[testinds]

np.set_printoptions(precision=3)
y = np.zeros(198)
yvar = np.ones(198)

# Tests
emuPCGPwM = emulator(inputs, thetatrain, np.copy(ftrain.T), method='PCGPwM',
               options={'xrmnan': 'all',
                        'thetarmnan': 'never',
                        'return_grad': True})

calPCGPwM = calibrator(emu=emuPCGPwM, y=y, yvar=yvar,
                       x=inputs, thetaprior=prior_fayans,
                       method='directbayeswoodbury',
                       args={'sampler': 'PTLMC'})

calPCGPwMpred = calPCGPwM.predict()
PCGPwMposttheta = calPCGPwM.theta.rnd(5000)
PCGPwM90CI = np.quantile(PCGPwMposttheta, q=(0.05, 0.95), axis=0).T

emuPCGPkNN = emulator(inputs, thetatrain, np.copy(ftrain.T), method='PCGPwImpute',
                      options={'xrmnan': 'all',
                               'thetarmnan': 'never',
                               'return_grad': True})
calPCGPkNN = calibrator(emu=emuPCGPkNN, y=y, yvar=yvar,
                        x=inputs, thetaprior=prior_fayans,
                        method='directbayeswoodbury',
                        args={'sampler': 'PTLMC'})

calPCGPkNNpred = calPCGPkNN.predict()
PCGPkNNposttheta = calPCGPkNN.theta.rnd(5000)
PCGPkNN90CI = np.quantile(PCGPkNNposttheta, q=(0.05, 0.95), axis=0).T

fcomplete = ftrain[~np.isnan(ftrain).any(1)]
thetacomplete = thetatrain[~np.isnan(ftrain).any(1)]
emuSimple = emulator(inputs, thetacomplete, np.copy(fcomplete.T), method='PCGPwImpute',
                     options={'xrmnan': 'all',
                              'thetarmnan': 'never',
                              'return_grad': True})

calSimple = calibrator(emu=emuSimple, y=y, yvar=yvar,
                       x=inputs, thetaprior=prior_fayans,
                       method='directbayeswoodbury',
                       args={'sampler': 'PTLMC'})

calSimplepred = calSimple.predict()
Simpleposttheta = calSimple.theta.rnd(5000)
Simple90CI = np.quantile(Simpleposttheta, q=(0.05, 0.95), axis=0).T

np.savetxt('PCGPkNNthetas.txt', PCGPkNNposttheta)
np.savetxt('PCGPwMposttheta.txt', PCGPwMposttheta)
np.savetxt('Simpleposttheta.txt', Simpleposttheta)

Priorwidth = sps.beta.ppf(0.95, 2, 2) - sps.beta.ppf(0.05, 2, 2)
Simplewidth = Simple90CI[:,1] - Simple90CI[:,0]
Simplepostmean = Simpleposttheta.mean(0)
PCGPwMwidth = PCGPwM90CI[:,1] - PCGPwM90CI[:,0]
PCGPwMpostmean = PCGPwMposttheta.mean(0)
PCGPkNNwidth = PCGPkNN90CI[:,1] - PCGPkNN90CI[:,0]
PCGPkNNpostmean = PCGPkNNposttheta.mean(0)

emuPCGPwMpred = emuPCGPwM.predict(inputs, thetatest)
emuPCGPkNNpred = emuPCGPkNN.predict(inputs, thetatest)
emuSimplepred = emuSimple.predict(inputs, thetatest)

emuPCGPwMmean = emuPCGPwMpred.mean()
emuPCGPkNNmean = emuPCGPkNNpred.mean()
emuSimplemean = emuSimplepred.mean()


rmse_x_PCGPwM = np.sqrt(np.nanmean((emuPCGPwMmean - ftest.T)**2, axis=1)) / np.nanmean(ftrain, axis=0)
rmse_x_PCGPkNN = np.sqrt(np.nanmean((emuPCGPkNNmean - ftest.T)**2, axis=1)) / np.nanmean(ftrain, axis=0)
rmse_x_Simple = np.sqrt(np.nanmean((emuSimplemean - ftest.T)**2, axis=1)) / np.nanmean(ftrain, axis=0)


import matplotlib.pyplot as plt
plt.style.use(['science','high-vis','grid'])
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(np.arange(198)+1, rmse_x_Simple, s=20, marker='x', label='complete data')
plt.scatter(np.arange(198)+1, rmse_x_PCGPwM, s=20, marker='D', label='PCGPwM')
plt.scatter(np.arange(198)+1, rmse_x_PCGPkNN, s=20, marker='o', label='PCGP-kNN')
ax.tick_params('both', labelsize=15)
plt.yscale('log')
plt.xlabel('Observables',fontsize=20)
plt.ylabel('RMSE',fontsize=20)
plt.tight_layout()
# plt.close()