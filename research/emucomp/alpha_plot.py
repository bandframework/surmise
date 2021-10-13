import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science', 'high-vis', 'grid'])

v = np.arange(0, 1, 0.01)
linestyles = ['-', '--', '-.', ':']

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for k, i in enumerate([1, 1/2, 0.25, 0]):
    ax.plot(v, v/((1-v)**i), label=r'$\alpha={:.2f}$'.format(i), linewidth=2, linestyle=linestyles[k])
plt.yscale('log')
plt.xlabel(r'$w_{ik}$', fontsize=20)
plt.ylabel(r'$v_{ik}$', rotation=0, labelpad=15, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\alpha_v.png', dpi=150)
