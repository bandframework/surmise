import scipy.io as spio
import numpy as np

mat = spio.loadmat(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\startingpt\starting_points_test_info.mat')
bigmap = np.loadtxt(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\startingpt\errmap.txt',delimiter=',',dtype=int)

fvals = mat['Fhist']
errvals = mat['Errorhist']
thetavals = mat['X0mat'].T
obsvals = mat['fvals'].T
inputs = np.loadtxt(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\startingpt\inputdata.csv', delimiter=',', dtype=object)

toterr = errvals @ bigmap
errvalssimple = toterr > 0.5

fvals[errvalssimple] = np.nan
sortinds = np.argsort(errvalssimple.sum(1))

fvals500 = fvals[:500]
sortinds500 = np.argsort(errvalssimple[:500].sum(1))
fvals500fail = errvalssimple[:500].mean()
import matplotlib.pyplot as plt
plt.style.use(['science', 'high-vis'])
fig, ax = plt.subplots()
plt.imshow(~np.isnan(fvals500[sortinds500]), aspect='auto', cmap='gray', interpolation='none')
plt.xlabel('observables', fontsize=15)
plt.ylabel('parameters', fontsize=15)
ax.tick_params(which='both', width=0)
plt.tight_layout()
dir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\startingpt'
plt.savefig(dir + r'\fayans_startingpt_fail.png', dpi=150)
plt.close()

# # Pair Plots
# import seaborn as sns
# import pandas as pd
# fayans_cols = [r'$\rho_{\mathrm{eq}}$', r'$E/A$', r'$K$', r'$J$',
#                     r'$L$', '$h^{\mathrm{v}}_{2{-}}$',
#                     r'$a^{\mathrm{s}}_{+}$',
#                     r'$h^{\mathrm{s}}_{\nabla}$',
#                     r'$\kappa$', r'$\kappa^\prime$',
#                     r'$f^{\xi}_{\mathrm{ex}}$',
#                     r'$h^{\xi}_{+}$', r'$h^{\xi}_{\nabla}$']
#
# theta_df = pd.DataFrame(thetavals[:100,], columns=fayans_cols)
# g = sns.PairGrid(theta_df, height=1,
#                  layout_pad=0.005,
#                  diag_sharey=True)
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.scatterplot, markers='x', alpha=0.5)
# g.map_lower(sns.kdeplot, shade=True)
# g.set(xlim=(0.1, 0.9), ylim=(0.1,0.9))
# # g.map_upper(sns.scatterplot, alpha=0.75)
