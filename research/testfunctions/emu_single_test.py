import json
import numpy as np
import time
from surmise.emulation import emulator
from testdiagnostics import errors


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


def single_test(emuname, x, theta, f, model, testtheta, modelname, ntheta, fail_random, fail_level, bigM, j, directory):
    emuname_orig = emuname
    try:
        emutime0 = time.time()
        if emuname == 'PCGPwM':
            withgrad = True
            args = {'epsilon': 0.001,
                    'lognugmean': -15,
                    'lognugLB': -22,
                    'bigM': bigM}
        elif emuname == 'PCGP_KNN':
            emuname = 'PCGPwMatComp'
            args = {'epsilon': 0.001,
                    'lognugmean': -15,
                    'lognugLB': -22,
                    'compmethod': 'KNN'}
            withgrad = True
        elif emuname == 'PCGP_BR':
            emuname = 'PCGPwMatComp'
            args = {'epsilon': 0.001,
                    'lognugmean': -15,
                    'lognugLB': -22,
                    'compmethod': 'BayesianRidge'}
            withgrad = True
        else:
            args = {}
            withgrad = False

        emu = emulator(x, theta, np.copy(f), method=emuname,
                       args=args,
                       options={'xrmnan': 'all',
                                'thetarmnan': 'never',
                                'return_grad': withgrad})
        emutime1 = time.time()

        res = errors(x, testtheta, model, modelname, fail_random, fail_level, bigM,
                     failfraction=(np.isnan(f).sum() / f.size),
                     ntheta=ntheta,
                     emu=emu,
                     emutime=emutime1-emutime0,
                     method=emuname_orig)

    except Exception as e:
        print(e)
        res = errors(x, testtheta, model, modelname, fail_random, fail_level, bigM,
                     failfraction=(np.isnan(f).sum() / f.size),
                     ntheta=ntheta,
                     emu=None,
                     emutime=None,
                     method=emuname_orig)

    dumper = json.dumps(res, cls=NumpyEncoder)
    fname = directory+r'\{:s}_{:s}_{:d}_rand{:s}{:s}_bigM{:d}_rep{:d}.json'.format(
            emuname_orig, modelname, ntheta, str(fail_random), fail_level, bigM, j)
    with open(fname, 'w') as fn:
        json.dump(dumper, fn)

    return fname
