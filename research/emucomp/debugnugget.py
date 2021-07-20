import numpy as np
import scipy.stats as sps
from boreholetestfunctions import borehole_model
from surmise.emulation import emulator
import time
import json
from pyDOE import lhs


#%% prior class
class thetaprior:
    """Prior class."""

    def lpdf(theta):
        """Return log density."""
        return (np.sum(sps.beta.logpdf(theta, 2, 2), 1)).reshape((len(theta), 1))

    def rnd(n):
        """Return random variables from the density."""
        return np.vstack((sps.beta.rvs(2, 2, size=(n, 4))))


def maintest(x, theta, emuname):
    f = borehole_model(x, theta)
    try:
        if emuname == 'PCGPwM':
            withgrad = True
        else:
            withgrad = False

        emu = emulator(x, theta, np.copy(f), method=emuname,
                       args={'epsilon': 0.0,
                             'lognugmean': -18,
                             'lognugLB': -24},
                       options={'xrmnan': 'all',
                                'thetarmnan': 'never',
                                'return_grad': withgrad})

        def mse(emu, x, theta, f):
            s = np.mean((emu.predict(x, theta).mean() - f) ** 2)
            return s
        return mse(emu, x, theta, f)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    # number of locations
    nx = 5
    x = sps.uniform.rvs(0, 1, (nx, 3))
    x[:, 2] = x[:, 2] > 0.5

    # number of parameters
    ntheta = 50
    thetas = np.zeros((3, ntheta, 4))
    np.random.seed(0)
    for i in np.arange(3):
        thetas[i] = lhs(4, ntheta)
    np.random.seed()

    # observations
    # yvar = 0.1 * np.ones(nx)
    # y = np.squeeze(borehole_true(x)) + sps.norm.rvs(0, np.sqrt(yvar))

    result = np.zeros((3, 3))
    for i in np.arange(3):
        print('RUN#{:d}'.format(i))
        # result[i, 0] = "{:.3E}".format(maintest(x, thetas[i], 'GPy'))
        # result[i, 1] = "{:.3E}".format(maintest(x, thetas[i], 'PCGP'))
        result[i, 2] = "{:.3E}".format(maintest(x, thetas[i], 'PCGPwM'))

    np.set_printoptions(precision=3)
    print('MSE (Training)')
    print('\t GPy \t PCGP \t PCGPwM')
    print(result)
    # print('Average MSE over 10 macro-replications')
    # print(result.mean(0))