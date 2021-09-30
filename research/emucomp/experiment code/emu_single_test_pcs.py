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


def single_test(emuname, x, theta, f, model, testtheta, modelname, ntheta,
                fail_random, fail_level, bigM, j, directory,
                skip_std=False, caller=None):
    emuname_orig = emuname
    try:
        epsilon = 0.0000001
        if skip_std:
            if caller is None:
                offset = np.nanmean(f, 1)
                scale = np.nanstd(f, 1)
                scale[scale == 0] = 0.0001
                fs = ((f.T - offset) / scale).T
                fs_comp = fs.copy()
                from sklearn import impute
                imputer = impute.KNNImputer()
                fs_comp = imputer.fit_transform(fs_comp)
                # fs_comp[np.isnan(fs)] = np.nanmean(fs_comp)
            else:
                f_comp = caller['nofailmodel'](x, theta)
                offset = np.mean(f_comp, 1)
                scale = np.std(f_comp, 1)
                fs_comp = ((f_comp.T - offset) / scale).T
                fs = fs_comp.copy()
                fs[np.isnan(f)] = np.nan

            U, S, _ = np.linalg.svd(fs_comp, full_matrices=False)
            Sp = S**2 - epsilon
            Up = U[:, Sp > 0]
            extravar = np.nanmean((fs_comp.T - fs_comp.T @ Up @ Up.T) ** 2, 0) * (scale ** 2)

            standardpcinfo = {'offset': offset,
                              'scale': scale,
                              'fs': fs.T,
                              'U': U,
                              'S': S,
                              'extravar': extravar
                              }
        else:
            standardpcinfo = None

        emutime0 = time.time()
        if emuname == 'PCGPwM':
            withgrad = True
            args = {'epsilon': epsilon,
                    'lognugmean': -15,
                    'lognugLB': -22,
                    'bigM': bigM,
                    'standardpcinfo': standardpcinfo}
        elif emuname == 'PCGP_KNN':
            emuname = 'PCGPwMatComp'
            args = {'epsilon': epsilon,
                    'lognugmean': -15,
                    'lognugLB': -22,
                    'compmethod': 'KNN'}
            withgrad = True
        elif emuname == 'PCGP_BR':
            emuname = 'PCGPwMatComp'
            args = {'epsilon': epsilon,
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

        print(skip_std, caller)
        print(res)
        return emu
    except Exception as e:
        print(e)
        res = errors(x, testtheta, model, modelname, fail_random, fail_level, bigM,
                     failfraction=(np.isnan(f).sum() / f.size),
                     ntheta=ntheta,
                     emu=None,
                     emutime=None,
                     method=emuname_orig)

    dumper = json.dumps(res, cls=NumpyEncoder)
    # fname = directory+r'\{:s}_{:s}_{:d}_rand{:s}{:s}_bigM{:d}_rep{:d}.json'.format(
    #         emuname_orig, modelname, ntheta, str(fail_random), fail_level, bigM, j)
    # with open(fname, 'w') as fn:
    #     json.dump(dumper, fn)

    return emu
