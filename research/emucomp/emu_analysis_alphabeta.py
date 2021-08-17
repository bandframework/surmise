import time

import numpy as np
import scipy.stats as sps
from surmise.emulation import emulator
from pyDOE import lhs
from testdiagnostics import plot_fails, plot_marginal, errors
import pandas as pd
import json


if __name__ == '__main__':
    # if failures are random
    random = True
    dampalphas = [0, 1/8, 1/4, 1/3, 1/2, 3/4, 1]

    error_results = []
    for k in np.arange(5):
        for i in np.arange(0, 4):
            if i == 0:
                import boreholetestfunctions as func
                from boreholetestfunctions import borehole_failmodel as failmodel
                from boreholetestfunctions import borehole_failmodel_random as failmodel_random
                from boreholetestfunctions import borehole_model as nofailmodel
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
            nx = 25
            x = sps.uniform.rvs(0, 1, (nx, xdim))
            np.random.seed()

            # number of parameters
            ntheta = 100
            origtheta = lhs(thetadim, ntheta)
            testtheta = np.random.uniform(0, 1, (1000, thetadim))

            # number of training parameters
            n = 50

            # whether model can fail
            model = failmodel if not random else failmodel_random

            theta = np.copy(origtheta[0:n])

            f = model(x, theta)

            # emuPCGPwM = emulator(x, theta, np.copy(f), method='PCGPwM',
            #                      options={'xrmnan': 'all',
            #                               'thetarmnan': 'never',
            #                               'return_grad': True})

            for j in np.arange(len(dampalphas)):
                alpha = dampalphas[j]

                emuPCGPwM = emulator(x, theta, np.copy(f), method='PCGPwM',
                                       args={'epsilon': 0.001,
                                             'lognugmean': -15,
                                             'lognugLB': -22,
                                             'dampalpha': alpha},
                                       options={'xrmnan': 'all',
                                                'thetarmnan': 'never',
                                                'return_grad': True})
                #
                # emuPCGPwMtoosmall =  emulator(x, theta, np.copy(f), method='PCGPwM',
                #                        args={'epsilon': 0.001,
                #                              'varconstant': np.exp(-12),
                #                              'lognugmean': -15,
                #                              'lognugLB': -22,
                #                              'dampalpha': alpha},
                #                        options={'xrmnan': 'all',
                #                                 'thetarmnan': 'never',
                #                                 'return_grad': True})
                #
                # emuPCGPwMsmall =  emulator(x, theta, np.copy(f), method='PCGPwM',
                #                        args={'epsilon': 0.001,
                #                              'varconstant': np.exp(-4),
                #                              'lognugmean': -15,
                #                              'lognugLB': -22,
                #                              'dampalpha': alpha},
                #                        options={'xrmnan': 'all',
                #                                 'thetarmnan': 'never',
                #                                 'return_grad': True})

                emuPCGPwMmed = emulator(x, theta, np.copy(f), method='PCGPwM',
                                       args={'epsilon': 0.001,
                                             'varconstant': np.exp(0),
                                             'lognugmean': -15,
                                             'lognugLB': -22,
                                             'dampalpha': alpha},
                                       options={'xrmnan': 'all',
                                                'thetarmnan': 'never',
                                                'return_grad': True})
                #
                # emuPCGPwMbig = emulator(x, theta, np.copy(f), method='PCGPwM',
                #                        args={'epsilon': 0.001,
                #                              'varconstant': np.exp(4),
                #                              'lognugmean': -15,
                #                              'lognugLB': -22,
                #                              'dampalpha': alpha},
                #                        options={'xrmnan': 'all',
                #                                 'thetarmnan': 'never',
                #                                 'return_grad': True})
                #
                #
                # emuPCGPwMtoobig = emulator(x, theta, np.copy(f), method='PCGPwM',
                #                        args={'epsilon': 0.001,
                #                              'varconstant': np.exp(20),
                #                              'lognugmean': -15,
                #                              'lognugLB': -22,
                #                              'dampalpha': alpha},
                #                        options={'xrmnan': 'all',
                #                                 'thetarmnan': 'never',
                #                                 'return_grad': True})

                print('Gpy now')
                #
                # emuGPy = emulator(x, theta, np.copy(f), method='GPy',
                #                   options={'xrmnan': 'all',
                #                            'thetarmnan': 'never'})


                # test theta
                print('\n Test errors')
                for emu in [emuPCGPwM, emuPCGPwMmed]: # [emuGPy, emuPCGPwM, emuPCGPwMtoosmall, emuPCGPwMsmall, emuPCGPwMmed, emuPCGPwMbig, emuPCGPwMtoobig]: #, emuPCGPwM1, emuPCGPwM2, emuPCGPwM5]:
                    d = errors(emu, x, testtheta, model, modelname, random)
                    error_results.append(d)
                    print(d)

            # plot_fails(x, testtheta, model)

            # plot_marginal(x, testtheta, model, modelname, [emuGPy, emuPCGPwM, emuPCGPwMtoosmall, emuPCGPwMbig, emuPCGPwMtoobig]) # , emuPCGPwMtoosmall, emuPCGPwMtoobig]) # emuPCGPwMtoosmall,

    error_df = pd.DataFrame(error_results)
    df = error_df.to_json()

    dirname = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\results'
    fname = r'\errors_{:s}_random{:s}.json'.format(time.strftime('%Y%m%d%H%M%S'), str(random))
    with open(dirname+fname, 'w') as f:
        json.dump(df, f)
