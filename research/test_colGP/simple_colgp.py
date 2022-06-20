import numpy as np
from surmise.emulation import emulator

def balldropmodel_grav(x, theta):
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1]
        g = theta[k]
        f[k, :] = h0 - (g / 2) * (t ** 2)
    return f.T


if __name__ == '__main__':
    x0 = np.arange(10) + 1
    x1 = np.random.gamma(10, 1, 10)
    x = np.vstack((x0, x1)).T
    theta = np.random.gamma(5, 5, 100).reshape((100, 1))
    f = balldropmodel_grav(x, theta)

    mask = np.random.uniform(0, 1, f.shape) > 0.9
    f_mis = f.copy()
    f_mis[mask] = np.nan

    emu = emulator(x=x, theta=theta, f=f, method='colGP')
    emupred = emu.predict()

    emu_mis = emulator(x=x, theta=theta, f=f_mis, method='colGP')
    emupred_mis = emu_mis.predict()