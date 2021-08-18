import time
import numpy as np
import scipy.stats as sps
from surmise.emulation import emulator
from pyDOE import lhs
from testdiagnostics import plot_fails, plot_marginal, errors
from multiprocessing import Process
import pandas as pd
import json


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


def maintest(emuname, x, theta, f, model, modelname, ntheta, random, j):
    try:
        emutime0 = time.time()

        if emuname == 'PCGPwM':
            withgrad = True
            args = {'epsilon': 0.001,
                    'lognugmean': -15,
                    'lognugLB': -22}
        elif emuname == 'PCGPwMatComp':
            args = {'epsilon': 0.001}
            withgrad = False
        else:
            args = {}
            withgrad = False

        emu = emulator(x, theta, np.copy(f), method=emuname,
                       args=args,
                       options={'xrmnan': 'all',
                                'thetarmnan': 'never',
                                'return_grad': withgrad})
        emutime1 = time.time()

        res = errors(x, theta, model, modelname, random,
                     ntheta=ntheta,
                     emu=emu,
                     emutime=emutime1-emutime0,
                     method=emuname)

    except Exception as e:
        print(e)
        res = errors(x, theta, model, modelname, random,
                     ntheta=ntheta,
                     emu=None,
                     emutime=None,
                     method=emuname)

    dumper = json.dumps(res, cls=NumpyEncoder)
    directory = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\emucomparison'
    with open(directory+r'\{:s}_{:s}_{:d}_rep{:d}.json'.format(emuname, modelname, n, j), 'w') as fn:
        json.dump(dumper, fn)


if __name__ == '__main__':
    # if failures are random
    random = True
    error_results = []
    ns = [10, 25, 50, 100, 250, 500]

    timeout = 900
    for j in np.arange(5):
        for i in np.arange(0, 4):
            if i == 0:
                import boreholetestfunctions as func
                from boreholetestfunctions import borehole_failmodel as failmodel
                from boreholetestfunctions import borehole_failmodel_random as failmodel_random
                from boreholetestfunctions import borehole_model as nofailmodel
                from boreholetestfunctions import borehole_true as truemodel
            elif i == 1:
                import TestingfunctionPiston as func
                from TestingfunctionPiston import Piston_failmodel as failmodel
                from TestingfunctionPiston import Piston_failmodel_random as failmodel_random
                from TestingfunctionPiston import Piston_model as nofailmodel
                from TestingfunctionPiston import Piston_true as truemodel
            elif i == 2:
                import TestingfunctionOTLcircuit as func
                from TestingfunctionOTLcircuit import OTLcircuit_failmodel as failmodel
                from TestingfunctionOTLcircuit import OTLcircuit_failmodel_random as failmodel_random
                from TestingfunctionOTLcircuit import OTLcircuit_model as nofailmodel
                from TestingfunctionOTLcircuit import OTLcircuit_true as truemodel
            elif i == 3:
                import TestingfunctionWingweight as func
                from TestingfunctionWingweight import Wingweight_failmodel as failmodel
                from TestingfunctionWingweight import Wingweight_failmodel_random as failmodel_random
                from TestingfunctionWingweight import Wingweight_model as nofailmodel
                from TestingfunctionWingweight import Wingweight_true as truemodel

            meta = func._dict
            modelname = meta['function']
            xdim = meta['xdim']
            thetadim = meta['thetadim']

            # number of locations
            np.random.seed(10)
            nx = 25
            x = sps.uniform.rvs(0, 1, (nx, xdim))
            np.random.seed()

            # number of parameters
            ntheta = 500
            origtheta = lhs(thetadim, ntheta)
            testtheta = np.random.uniform(0, 1, (1000, thetadim))
            for k in np.arange(len(ns)):
                # number of training parameters
                n = ns[k]
                # whether model can fail
                model = failmodel if not random else failmodel_random
                theta = np.copy(origtheta[0:n])
                f = model(x, theta)

                proc_GPy = Process(target=maintest('GPy', x, theta, f, model, modelname, n, random, j), name='Process_GPy_{:d}_{:s}_random{:s}'.format(n, modelname, str(random)))
                proc_PCGPwM = Process(target=maintest('PCGPwM', x, theta, f, model, modelname, n, random, j), name='Process_PCGPwM_{:d}_{:s}_random{:s}'.format(n, modelname, str(random)))
                proc_PCGPwMC = Process(target=maintest('PCGPwMatComp', x, theta, f, model, modelname, n, random, j), name='Process_PCGPwMatComp_{:d}_{:s}_random{:s}'.format(n, modelname, str(random)))

                proc_GPy.start()
                proc_PCGPwM.start()
                proc_PCGPwMC.start()

                proc_GPy.join(timeout=timeout)
                proc_PCGPwM.join(timeout=timeout)
                proc_PCGPwMC.join(timeout=timeout)

                proc_GPy.terminate()
                proc_PCGPwM.terminate()
                proc_PCGPwMC.terminate()