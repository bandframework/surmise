import os.path
import time
import numpy as np
import scipy.stats as sps
from inspect import getsourcefile
from os.path import abspath, dirname, basename, normpath
from pathlib import Path
from glob import glob
from testfunc_wrapper import TestFunc
from emu_single_test import single_test, single_test_fayans


def make_dirs():
    current_dir = abspath(dirname(getsourcefile(lambda: 0)))  # where this script is
    results_dir = r'\emulator_timing_results'
    parent_dir = current_dir + results_dir
    if not os.path.exists(parent_dir):
        Path(parent_dir).mkdir(exist_ok=True)
    subdirlist = [basename(normpath(x)) for x in glob(parent_dir+'\\*\\')]
    if len(subdirlist) < 0.5:
        directory = parent_dir + r'\0'
    else:
        maxint = max(np.array(subdirlist).astype(int))
        directory = (parent_dir + r'\{:d}').format(maxint + 1)

    data_dir = directory + r'\data'
    plot_dir = directory + r'\plot'
    Path(directory).mkdir(exist_ok=True)
    Path(data_dir).mkdir(exist_ok=True)
    Path(plot_dir).mkdir(exist_ok=True)

    return data_dir, plot_dir


def run_experiment(data_dir):
    # Macro replication
    nrep = 1
    js = np.arange(nrep)

    # Number of input locations
    nx = 200
    # Number of parameters
    ns = [500] #, 2500]

    # Knobs options
    fail_configs = [
                    # (True, 0.01),
                    # (True, 0.05),
                    # (True, 0.25),
                    # (False, 0.01),
                    # (False, 0.05),
                    (False, 0.25),
                    ]
    models = ['borehole'] #, 'piston', 'otlcircuit', 'wingweight'] # 'borehole',
    emulator_methods = ['EMGP'] #, 'PCGP_KNN']  # 'GPy' #


    # JSON filelist
    totalruns = len(js) * len(ns) * len(fail_configs) * len(emulator_methods) * len(models)
    resultJSONs = []

    for func in models:
        # Query test function for Borehole
        func_caller = TestFunc(func).info
        function_name = func_caller['function']
        xdim = func_caller['xdim']
        thetadim = func_caller['thetadim']

        thetasampler = sps.qmc.LatinHypercube(d=thetadim)
        for j in js:
            x = sps.uniform.rvs(0, 1, (nx, xdim))
            testtheta = np.random.uniform(0, 1, (1000, thetadim))

            for n in ns:
                theta = thetasampler.random(n)
                for fail_random, fail_level in fail_configs:
                    if fail_level == 'none':
                        model = func_caller['nofailmodel']
                        f = model(x, theta)
                    elif fail_random is True:
                        model = func_caller['failmodel_random']
                        f = model(x, theta, fail_level)
                    elif fail_random is False:
                        model = func_caller['failmodel']
                        f = model(x, theta, fail_level)
                    else:
                        raise ValueError('Invalid failures configuration.')

                    for method in emulator_methods:
                        print('{:d} of {:d} runs completed'.format(len(resultJSONs), totalruns))
                        result_fname = single_test(method, x, theta, f, model, testtheta,
                                                   function_name, n, fail_random, fail_level,
                                                   j, data_dir, func_caller)
                        resultJSONs.append(result_fname)
                        # if divmod(len(resultJSONs), 10) == 0:
                        print(result_fname)
    return resultJSONs


def run_fayans_local(data_dir):
    class prior_fayans:
        def lpdf(theta):
            return sps.beta.logpdf(theta, 2, 2).sum(1).reshape((theta.shape[0], 1))
        def rnd(n):
            return sps.beta.rvs(2, 2, size=(n, 13))

    import scipy.io as spio
    mat = spio.loadmat(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\startingpt\starting_points_test_info.mat')
    bigmap = np.loadtxt(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\startingpt\errmap.txt', delimiter=',', dtype=int)
    inputs = np.loadtxt(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\startingpt\inputdata.csv', delimiter=',', dtype=object)

    xval = inputs[:, :-1].astype(float)
    xvalmean = xval.mean(0)
    xvalstd = xval.std(0)
    xvalrng = np.max(xval, 0) - np.min(xval, 0)
    xval = (xval - xvalmean) / xvalrng
    xcat = inputs[:, -1]

    uniquecat = np.unique(xcat)
    xcatnum = np.zeros((xcat.shape[0], uniquecat.shape[0]-1))
    for i in range(uniquecat.shape[0]-1):
        xcatnum[:, i][xcat == uniquecat[i]] = 1


    # inputs = np.column_stack((xval, xcatnum))

    fvals = mat['Fhist']
    errvals = mat['Errorhist']
    thetavals = mat['X0mat'].T
    obsvals = mat['fvals'].T
    toterr = errvals @ bigmap
    errvalssimple = toterr > 0.5

    fvals[errvalssimple] = np.nan

    n = 50  ## CHANGE
    thetanorm = np.linalg.norm(thetavals - 0.5, ord=1, axis=1)
    thetatopinds = np.argpartition(thetanorm, -n)[-n:]

    # subset training data
    ftrain = fvals[thetatopinds]
    thetatrain = thetavals[thetatopinds]
    testinds = np.setdiff1d(np.arange(1000), thetatopinds)  ## CHANGE
    ftest = fvals[testinds]
    thetatest = thetavals[testinds]

    np.set_printoptions(precision=3)
    y = np.zeros(198)
    yvar = np.ones(198)

    emulator_methods = ['EMGP'] #, 'PCGPwM', 'PCGP_KNN', 'colGP']  # 'GPy' #'PCGPwM', 'PCGP_KNN',

    # JSON filelist
    totalruns = len(emulator_methods)
    resultFayansJSONs = []

    for method in emulator_methods:
        print(method)
        print('{:d} of {:d} runs completed'.format(len(resultFayansJSONs), totalruns))
        result_fname = single_test_fayans(method, inputs, thetatrain, ftrain.T, thetatest,
                                          ftest.T, y, yvar, prior_fayans=prior_fayans, directory=data_dir)

        resultFayansJSONs.append(result_fname)
        print(result_fname)
    return resultFayansJSONs


if __name__ == '__main__':
    # data_dir, plot_dir = make_dirs()
    #
    # start_time = time.time()
    # listJSONs = run_experiment(data_dir)
    # run_time = time.time() - start_time
    # print('total runtime: {:.3f} seconds'.format(run_time))

    import warnings
    warnings.simplefilter('error')
    data_fay_dir, plot_fay_dir = make_dirs()
    listJSONs = run_fayans_local(data_fay_dir)