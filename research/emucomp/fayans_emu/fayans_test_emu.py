import scipy.io
import numpy as np
from surmise.emulation import emulator

mat = scipy.io.loadmat(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emu\FayansFall2021.mat')
errmap = np.loadtxt(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emu\errmap.txt', delimiter=',', dtype=int)

esave = mat['ESAVE']
theta = mat['XSAVE']
f = mat['FSAVE']
inputs = np.loadtxt(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emu\inputdata.csv', delimiter=',', dtype=object)

toterr = esave @ errmap
simpleerr = (toterr > 0.5)
wherefails = np.argwhere(simpleerr)
for r, c in wherefails:
    f[r, c] = np.nan

completeinds = np.argwhere(simpleerr.sum(1) < 0.5).squeeze()
incompleteinds = np.argwhere(simpleerr.sum(1) > 0.5).squeeze()

# compile data
train_inds = np.hstack((np.random.choice(completeinds, 400, replace=False),
                     np.random.choice(incompleteinds, 100, replace=False)))
ftrain = f[train_inds]
thetatrain = theta[train_inds]
test_inds = np.setdiff1d(np.arange(f.shape[0]), train_inds)
ftest = f[test_inds]
thetatest = theta[test_inds]

import time
start = time.time()
emu = emulator(inputs, thetatrain, np.copy(ftrain), method='PCGPwM',
               options={'xrmnan': 'all',
                        'thetarmnan': 'never',
                        'return_grad': True})
end = time.time()
print('time taken: ', end - start)

pred = emu.predict(inputs, thetatest)
predmean = pred.mean()

mse = (predmean - ftest.T)**2
frng = np.atleast_2d(np.nanmax(ftest, 0) - np.nanmin(ftest, 0)).T
rmse_x = np.sqrt(np.nanmean((predmean - ftest.T)**2 / frng, axis=1))
rmse_theta = np.sqrt(np.nanmean((predmean - ftest.T)**2 / frng, axis=0))

import matplotlib.pyplot as plt
plt.style.use(['science','high-vis','grid'])
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(np.arange(198)+1, rmse_x, marker='x')
ax.tick_params('both', labelsize=15)
plt.yscale('log')
plt.xlabel('Observable',fontsize=20)
plt.ylabel('RMSE',fontsize=20)
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(6,6))
plt.imshow(np.isnan(ftrain), aspect='auto', cmap='Reds', interpolation='none')
ax.tick_params('both', labelsize=15)
plt.ylabel('parameters', fontsize=20)
plt.xlabel('observables', fontsize=20)
