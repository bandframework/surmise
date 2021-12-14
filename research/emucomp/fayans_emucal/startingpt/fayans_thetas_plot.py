import numpy as np
import pandas as pd

import os
os.chdir(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\startingpt')

fayans_cols = [r'$\rho_{\mathrm{eq}}$', r'$E/A$', r'$K$', r'$J$',
               r'$L$', '$h^{\mathrm{v}}_{2{-}}$',
               r'$a^{\mathrm{s}}_{+}$',
               r'$h^{\mathrm{s}}_{\nabla}$',
               r'$\kappa$', r'$\kappa^\prime$',
               r'$f^{\xi}_{\mathrm{ex}}$',
               r'$h^{\xi}_{+}$', r'$h^{\xi}_{\nabla}$',
               'emu']

PCGPkNNpostthetas = np.loadtxt('PCGPkNNthetas.txt')
PCGPwMpostthetas = np.loadtxt('PCGPwMposttheta.txt')
Simplepostthetas = np.loadtxt('Simpleposttheta.txt')

ciPCGPwM = np.array((-1, 1)) @ np.quantile(PCGPwMpostthetas, (0.05, 0.95), axis=0)
ciPCGPkNN = np.array((-1, 1)) @ np.quantile(PCGPkNNpostthetas, (0.05, 0.95), axis=0)
ciSimple = np.array((-1, 1)) @ np.quantile(Simplepostthetas, (0.05, 0.95), axis=0)

dfwM = pd.DataFrame(PCGPwMpostthetas, columns=fayans_cols[:-1])
dfkNN = pd.DataFrame(PCGPkNNpostthetas, columns=fayans_cols[:-1])
dfSimple = pd.DataFrame(Simplepostthetas, columns=fayans_cols[:-1])

dfwM['emu'] = 'PCGPwM'
dfkNN['emu'] = 'PCGP-kNN'
dfSimple['emu'] = 'Complete data'

df = pd.concat((dfwM, dfkNN, dfSimple))

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['science', 'high-vis'])
sns.boxplot(x='variable', y='value', hue='emu', whis=1, data=pd.melt(df, id_vars='emu'))

