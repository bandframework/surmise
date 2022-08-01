import time

import numpy as np
import scipy.stats as sps
from surmise.emulation import emulator
from testdiagnostics import errors
import pandas as pd
import json


if __name__ == '__main__':
    # if failures are random
    random = True
    dampalphas = [0, 1/4, 1/2, 3/4, 1]

    error_results = []
    for k in np.arange(3):
        for i in np.arange(0, 4):
            if i == 0:
                import TestingfunctionBorehole as func
                from TestingfunctionBorehole import borehole_failmodel as failmodel
                from TestingfunctionBorehole import borehole_failmodel_random as failmodel_random
                from TestingfunctionBorehole import borehole_model as nofailmodel
            elif i == 1:
                import TestingfunctionPiston as func
                from TestingfunctionPiston import Piston_failmodel as failmodel
                from TestingfunctionPiston import Piston_failmodel_random as failmodel_random
                from TestingfunctionPiston import Piston_model as nofailmodel
            elif i == 2:
                import TestingfunctionOTLcircuit as func
                from TestingfunctionOTLcircuit import OTLcircuit_failmodel as failmodel
                from TestingfunctionOTLcircuit import OTLcircuit_failmodel_random as failmodel_random
                from TestingfunctionOTLcircuit import OTLcircuit_model as nofailmodel

            elif i == 3:
                import TestingfunctionWingweight as func
                from TestingfunctionWingweight import Wingweight_failmodel as failmodel
                from TestingfunctionWingweight import Wingweight_failmodel_random as failmodel_random
                from TestingfunctionWingweight import Wingweight_model as nofailmodel

            meta = func._dict
            modelname = meta['function']
            xdim = meta['xdim']
            thetadim = meta['thetadim']

            # number of locations
            np.random.seed(10)
            nx = 15
            x = sps.uniform.rvs(0, 1, (nx, xdim))
            np.random.seed()

            # number of parameters
            ntheta = 250
            sampler = sps.qmc.LatinHypercube(d=thetadim)
            theta = sampler.random(ntheta)
            testtheta = np.random.uniform(0, 1, (1000, thetadim))

            # whether model can fail
            model = failmodel_random
            f = model(x, theta, p=0.25)

            for j in np.arange(len(dampalphas)):
                alpha = dampalphas[j]

                emuPCGPwM = emulator(x, theta, np.copy(f), method='PCGPwM',
                                       args={'dampalpha': alpha,
                                             'epsilonImpute': 10e-8,
                                             'lognugmean': -15,
                                             'lognugLB': -22,
                                             },
                                       options={'xrmnan': 'all',
                                                'thetarmnan': 'never',
                                                'return_grad': True})
                emuPCGPwMfixedb = emulator(x, theta, np.copy(f), method='PCGPwM',
                                           args={'dampalpha': alpha,
                                                 'epsilonImpute': 10e-8,
                                                 'varconstant': 1,
                                                 'lognugmean': -15,
                                                 'lognugLB': -22,
                                                 },
                                           options={'xrmnan': 'all',
                                                    'thetarmnan': 'never',
                                                    'return_grad': True}
                                           )
                # test theta
                print('\n Test errors')
                for emu in [emuPCGPwM, emuPCGPwMfixedb]: # [emuGPy, emuPCGPwM, emuPCGPwMtoosmall, emuPCGPwMsmall, emuPCGPwMmed, emuPCGPwMbig, emuPCGPwMtoobig]: #, emuPCGPwM1, emuPCGPwM2, emuPCGPwM5]:
                    d = errors(x, testtheta, model, modelname, random, ntheta,
                               0.05, emu, emutime=0, method='PCGPwM')
                    error_results.append(d)
                    print(d)

    error_df = pd.DataFrame(error_results)
    df = error_df.to_json()

    dirname = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\results_alphabeta'
    fname = r'\errors_{:s}_random{:s}.json'.format(time.strftime('%Y%m%d%H%M%S'), str(random))
    with open(dirname+fname, 'w') as f:
        json.dump(df, f)
