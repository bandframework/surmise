import numpy as np
from surmise.emulation import emulator
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])

x = np.atleast_2d(np.arange(-np.pi, np.pi, np.pi/100)).T
w = np.atleast_2d(np.arange(1, 10, 2) ** 2).T
y = np.array([np.sin(w[i]*x[j]) for i in range(w.shape[0]) for j in range(x.shape[0])]).reshape((x.shape[0], w.shape[0]))


emu = emulator(x, w, y, 'PCGPwM')

w0 = np.atleast_2d((1, 5, 10))

pred = emu.predict(x, w0)
predmean = pred.mean()
