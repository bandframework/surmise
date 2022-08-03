import json
import numpy as np
import time
from surmise.emulation import emulator
from surmise.calibration import calibrator
from testdiagnostics import errors, errors_fayans, calresults_fayans #, errors_fayans_EMGPind


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
                fail_random, fail_frac, j, directory, caller):
    emuname_orig = emuname
    epsilonPC = 0.001
    if emuname == 'PCGP_benchmark':
        skip_std = True
        caller = caller
    else:
        skip_std = False
        caller = None
    try:
        standardpcinfo = None
        if skip_std:
            f_comp = caller['nofailmodel'](x, theta)
            offset = np.mean(f_comp, 1)
            scale = np.std(f_comp, 1)
            fs_comp = ((f_comp.T - offset) / scale).T
            fs = fs_comp.copy()
            fs[np.isnan(f)] = np.nan

            U, S, _ = np.linalg.svd(fs_comp, full_matrices=False)
            Sp = S ** 2 - epsilonPC
            Up = U[:, Sp > 0]
            extravar = np.nanmean((fs_comp.T - fs_comp.T @ Up @ Up.T) ** 2, 0) * (scale ** 2)

            standardpcinfo = {'offset': offset,
                              'scale': scale,
                              'fs': fs.T,
                              'U': U,
                              'S': S,
                              'extravar': extravar
                              }

        emutime0 = time.time()
        if emuname == 'PCGPwM':
            withgrad = True
            args = {'lognugmean': -15,
                    'lognugLB': -22,
                    'nmaxhyptrain': 1000}
        elif emuname == 'PCGP_KNN':
            emuname = 'PCGPwImpute'
            args = {'lognugmean': -15,
                    'lognugLB': -22,
                    'compmethod': 'KNN',
                    'nmaxhyptrain': 1000}
            withgrad = True
        elif emuname == 'PCGP_BR':
            emuname = 'PCGPwImpute'
            args = {'lognugmean': -15,
                    'lognugLB': -22,
                    'compmethod': 'BayesianRidge',
                    'nmaxhyptrain': 1000}
            withgrad = True
        elif emuname == 'PCGP_benchmark':
            emuname = 'PCGPwM'
            args = {'lognugmean': -15,
                    'lognugLB': -22,
                    'standardpcinfo': standardpcinfo,
                    'nmaxhyptrain': 1000}
            withgrad = True
        elif emuname == 'colGP':
            args = {}
            withgrad = False
        elif emuname == 'GPEmGibbs':
            args = {'cat': False}
            withgrad = False
        else:
            args = {}
            withgrad = False

        options = {'xrmnan': 'all',
                   'thetarmnan': 'never',
                   'return_grad': withgrad}
        emu = emulator(x, theta, np.copy(f), method=emuname,
                       args=args,
                       options=options)
        emutime1 = time.time()

        res = errors(x, testtheta, model, modelname, fail_random,
                     failfraction=fail_frac,
                     ntheta=ntheta,
                     emu=emu,
                     emutime=emutime1 - emutime0,
                     method=emuname_orig)

    except Exception as e:
        print(e)
        res = errors(x, testtheta, model, modelname, fail_random,
                     failfraction=fail_frac,
                     ntheta=ntheta,
                     emu=None,
                     emutime=None,
                     method=emuname_orig)

    dumper = json.dumps(res, cls=NumpyEncoder)
    fname = directory + r'/{:s}_{:s}_{:d}_rand{:s}{:s}_rep{:d}_{:d}.json'.format(
        emuname_orig, modelname, ntheta, str(fail_random), str(int(fail_frac * 100)), j, np.random.randint(1000, 99999))
    with open(fname, 'w') as fn:
        json.dump(dumper, fn)

    return fname


def single_test_fayans(emuname, x, theta, f, testtheta,
                       testf, y, yvar, prior_fayans, directory):
    emuname_orig = emuname
    modelname = 'fayans'
    try:
        emutime0 = time.time()
        if emuname == 'PCGPwM':
            withgrad = True
            args = {'lognugmean': -15,
                    'lognugLB': -22,
                    'nmaxhyptrain': 1000}
        elif emuname == 'PCGP_KNN':
            emuname = 'PCGPwImpute'
            args = {'lognugmean': -15,
                    'lognugLB': -22,
                    'compmethod': 'KNN',
                    'nmaxhyptrain': 1000}
            withgrad = True
        elif emuname == 'PCGP_BR':
            emuname = 'PCGPwImpute'
            args = {'lognugmean': -15,
                    'lognugLB': -22,
                    'compmethod': 'BayesianRidge',
                    'nmaxhyptrain': 1000}
            withgrad = True
        elif emuname == 'colGP':
            args = {}
            withgrad = False
        elif emuname == 'GPEmGibbs':
            args = {'cat': False}
            withgrad = True
        else:
            args = {}
            withgrad = False

        options = {'xrmnan': 'all',
                   'thetarmnan': 'never',
                   'return_grad': withgrad}

        # if not emuname == 'GPEmGibbs':
        emu = emulator(x, theta, np.copy(f), method=emuname,
                       args=args,
                       options=options)
        emutime1 = time.time()

        res = errors_fayans(x, testtheta, testf, modelname, theta.shape[0],
                            emu=emu,
                            emutime=emutime1 - emutime0,
                            method=emuname_orig)

        # else:
        #     emus = emulator_EMGPind(x=x, theta=theta, f=f, method=emuname,
        #                             args=args, options=options)
        #
        #     emutime1 = time.time()
        #
        #     res = errors_fayans_EMGPind(x, testtheta, testf, modelname, theta.shape[0],
        #                                 emu=emus,
        #                                 emutime=emutime1 - emutime0,
        #                                 method=emuname_orig)

        # cal = calibrator(emu=emu, y=y, yvar=yvar,
        #                x=x, thetaprior=prior_fayans,
        #                method='directbayeswoodbury',
        #                args={'sampler': 'PTLMC'})
        #
        # rescal = calresults_fayans(cal, emu, x, thetatest=testtheta, ftest=testf, ftrain=f)


    except Exception as e:
        print(e)
        res = errors_fayans(x, testtheta, testf, modelname, theta.shape[0],
                            emu=None,
                            emutime=None,
                            method=emuname_orig)
        rescal = {'CI90width': None,
                  'postmean': None}
    dumper = json.dumps(res, cls=NumpyEncoder)
    fname = directory + r'/{:s}_{:s}.json'.format(
        emuname_orig, modelname)
    with open(fname, 'w') as fn:
        json.dump(dumper, fn)

    # dumper_f = json.dumps(rescal, cls=NumpyEncoder)
    #
    # fayans_fname = directory + r'/{:s}_{:s}_cal.json'.format(
    #     emuname_orig, modelname)
    # with open(fayans_fname, 'w') as fn:
    #     json.dump(dumper_f, fn)

    return fname


def emulator_EMGPind(x, theta, f, method, args, options):
    xval = x[:, :-1].astype(float)
    xcat = x[:, -1]
    emulist = []

    uniquecat = np.unique(xcat)
    ncat = uniquecat.shape[0]

    for i in range(ncat):
        xi = xval[xcat == uniquecat[i]]
        fi = f[xcat == uniquecat[i]]

        emu = emulator(x=xi, f=fi, theta=theta, method=method, args=args, options=options)
        emulist.append(emu)

    print('complete emulation')
    return emulist
