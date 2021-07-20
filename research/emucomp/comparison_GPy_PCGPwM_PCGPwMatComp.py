import numpy as np
import scipy.stats as sps
from boreholetestfunctions import borehole_model, borehole_failmodel, borehole_true, borehole_failmodel_random
from surmise.emulation import emulator
from surmise.calibration import calibrator
import time
import json
from multiprocessing import Process
from pyDOE import lhs


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#%% prior class
class thetaprior:
    """Prior class."""

    def lpdf(theta):
        """Return log density."""
        return (np.sum(sps.beta.logpdf(theta, 2, 2), 1)).reshape((len(theta), 1))

    def rnd(n):
        """Return random variables from the density."""
        return np.vstack((sps.beta.rvs(2, 2, size=(n, 4))))


def maintest(n, emuname, calname='directbayes', fail=False):
    theta = np.copy(origtheta[0:n])

    model = borehole_failmodel if fail else borehole_model

    f = model(x, theta)
    fmask = ~np.isinf(f)

    results = {}
    try:
        emutime0 = time.time()

        if emuname == 'PCGPwM':
            withgrad = True
        else:
            withgrad = False
        emu = emulator(x, theta, np.copy(f), method=emuname,
                       options={'epsilon': 0.1,
                                'xrmnan': 'all',
                                'thetarmnan': 'never',
                                'return_grad': withgrad})

        def mse(emu, x, theta, f):
            return np.mean((emu.predict(x, theta).mean() - f) ** 2)

        print('MSE, with {:s}'.format(emuname), mse(emu, x, theta, f))
        return
        emutime1 = time.time()

        if emuname == 'PCGPwM' and calname == 'directbayes':
            calname = 'directbayeswoodbury'
        cal = calibrator(emu, y, x, thetaprior, yvar,
                        method=calname,
                        args={'sampler': 'PTLMC'})
        caltime1 = time.time()

        p = emu.predict(x, theta)
        predicttime1 = time.time()

        p_mean = p.mean()

        p_mse = np.mean((p_mean[fmask] - f[fmask]) ** 2)
        print('MSE = ', p_mse)

        thetas = cal.theta.rnd(10000)
        thetarng = np.quantile(thetas, (0.025, 0.5, 0.975), axis=0)

        print('estimated posterior quantile:\n', np.round(thetarng[(0, -1), :], 3))
        results['mse'] = p_mse
        results['emu_time'] = emutime1 - emutime0
        results['cal_time'] = caltime1 - emutime1
        results['pred_time'] = predicttime1 - caltime1
        results['quantiles'] = thetarng

    except Exception as e:
        print(e)
        results['mse'] = None
        results['emu_time'] = None
        results['cal_time'] = None
        results['pred_time'] = None
        results['quantiles'] = None

    dumper = json.dumps(results, cls=NumpyEncoder)
    with open('emucomparison/{:s}_{:s}_{:d}_fail{:s}.json'.format(emuname, calname, n, str(fail)), 'w') as f:
        json.dump(dumper, f)

    return


if __name__ == '__main__':
    # number of locations
    nx = 25
    x = sps.uniform.rvs(0, 1, (nx, 3))
    x[:, 2] = x[:, 2] > 0.5

    # number of parameters
    ntheta = 500
    origtheta = lhs(4, ntheta)

    # observations
    yvar = 0.1 * np.ones(nx)
    y = np.squeeze(borehole_true(x)) + sps.norm.rvs(0, np.sqrt(yvar))

    maintest(50, 'PCGP')
    maintest(50, 'PCGPwM')
    maintest(50, 'GPy')
    #
    # timeout = 3600
    # proc_log = {}
    # for n in [10, 25, 50, 100, 150, 200, 250]:
    #     if n > 50:
    #         break
    #     proc_GPy = Process(target=maintest(n, 'GPy'), name='Process_GPy_{:d}'.format(n))
    #     proc_PCGPwM = Process(target=maintest(n, 'PCGPwM'), name='Process_PCGPwM_{:d}'.format(n))
    #     proc_PCGPwMC = Process(target=maintest(n, 'PCGPwMatComp'), name='Process_PCGPwMatComp_{:d}'.format(n))
    #     proc_PCGPwM_simpost = Process(target=maintest(n, 'PCGPwM', 'simulationpost'), name='Process_PCGPwM_simpost_{:d}'.format(n))
    #
    #     proc_GPy.start()
    #     proc_PCGPwM.start()
    #     proc_PCGPwMC.start()
    #     proc_PCGPwM_simpost.start()
    #
    #     proc_GPy.join(timeout=timeout)
    #     proc_PCGPwM.join(timeout=timeout)
    #     proc_PCGPwMC.join(timeout=timeout)
    #     proc_PCGPwM_simpost.join(timeout=timeout)
    #
    #     proc_GPy.terminate()
    #     proc_PCGPwM.terminate()
    #     proc_PCGPwMC.terminate()
    #     proc_PCGPwM_simpost.terminate()
    #
    #     if proc_GPy.exitcode is None:
    #         proc_log[proc_GPy.name] = -1
    #     elif proc_GPy.exitcode == 0:
    #         proc_log[proc_GPy.name] = 0
    #     if proc_PCGPwM.exitcode is None:
    #         proc_log[proc_PCGPwM.name] = -1
    #     elif proc_PCGPwM.exitcode == 0:
    #         proc_log[proc_PCGPwM.name] = 0
    #     if proc_PCGPwMC.exitcode is None:
    #         proc_log[proc_PCGPwMC.name] = -1
    #     elif proc_PCGPwMC.exitcode == 0:
    #         proc_log[proc_PCGPwMC.name] = 0
    #     if proc_PCGPwM_simpost.exitcode is None:
    #         proc_log[proc_PCGPwM_simpost.name] = -1
    #     elif proc_PCGPwM_simpost.exitcode == 0:
    #         proc_log[proc_PCGPwM_simpost.name] = 0

    # pass_emu = emulator(x, passthroughfunc=borehole_model, method='PCGPwM',
    #                     options={'xrmnan': 'all',
    #                              'thetarmnan': 'never',
    #                              'return_grad': True})
    #
    # # apply emulator to calibration
    # true_cal = calibrator(pass_emu, y, x, thetaprior, yvar,
    #                       method='directbayeswoodbury',
    #                       args={'sampler': 'PTLMC'})
    # postthetas = true_cal.theta.rnd(10000)
    # postthetarng = np.quantile(postthetas, (0.025, 0.5, 0.975), axis=0)
    #
    # print('true posterior quantile:\n', np.round(postthetarng[(0, -1), :], 3))
