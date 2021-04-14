import numpy as np
import scipy.stats as sps
from boreholetestfunctions import borehole_model, borehole_failmodel, borehole_true, borehole_failmodel_random
from surmise.emulation import emulator
from surmise.calibration import calibrator


#%% prior class
class thetaprior:
    """Prior class."""

    def lpdf(theta):
        """Return log density."""
        return (np.sum(sps.beta.logpdf(theta, 2, 2), 1)).reshape((len(theta), 1))

    def rnd(n):
        """Return random variables from the density."""
        return np.vstack((sps.beta.rvs(2, 2, size=(n, 4))))


# %% end of run data
x = np.loadtxt('x.txt', delimiter=',')
theta = np.loadtxt('theta.txt', delimiter=',')
y = np.loadtxt('y.txt', delimiter=',')
yvar = np.loadtxt('yvar.txt', delimiter=',')

# %% true posterior
pass_emu = emulator(x, passthroughfunc=borehole_model, method='PCGPwM',
                    options={'xrmnan': 'all',
                             'thetarmnan': 'never',
                             'return_grad': True})

# apply emulator to calibration
true_cal = calibrator(pass_emu, y, x, thetaprior, yvar, method='directbayeswoodbury')
postthetas = true_cal.theta.rnd(10000)
postthetarng = np.quantile(postthetas, (0.025, 0.5, 0.975), axis=0)

# %% emulation calibration
emu = emulator(x, theta, borehole_model(x, theta), method='PCGPwM',
               options={'xrmnan': 'all',
                        'thetarmnan': 'never',
                        'return_grad': True})

cal = calibrator(emu, y, x, thetaprior, yvar, method='directbayeswoodbury')
sampthetas = cal.theta.rnd(10000)
sampthetarng = np.quantile(sampthetas, (0.025, 0.5, 0.975), axis=0)

#%% quantile comparisons
# print('estimated quantile from the last loop:\n', np.round(res['quantile'][-1][(0,-1),:], 3))
print('estimated posterior quantile:\n', np.round(sampthetarng[(0,-1),:], 3))
print('true posterior quantile:\n', np.round(postthetarng[(0,-1), :], 3))

print('\n', np.round(postthetarng[(0,-1), :] - sampthetarng[(0,-1),:], 3))
