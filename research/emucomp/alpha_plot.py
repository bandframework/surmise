import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
plt.rcParams["mathtext.fontset"] = "dejavuserif"

v = np.arange(0, 1, 0.01)
linestyles = ['-', '--', '-.', ':']

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for k, i in enumerate([1, 1/2, 1/4, 0]):
    ax.plot(v, v/((1-v)**i), label=r'$\alpha={:.2f}$'.format(i), linestyle=linestyles[k])
plt.yscale('log')
plt.xlabel(r'$w_{ik}$')
plt.ylabel(r'$v_{ik}$', rotation=0)
plt.legend()
plt.savefig(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\alpha_v.png', dpi=150)
