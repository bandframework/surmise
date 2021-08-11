import numpy as np
import matplotlib.pyplot as plt

v = np.arange(0, 1, 0.01)
fig, ax = plt.subplots(1, 1)
for i in [0, 1/2, 1/3, 1]:
    ax.plot(v, v/((1-v)**i), label=r'$\alpha={:.2f}$'.format(i))
plt.yscale('log')