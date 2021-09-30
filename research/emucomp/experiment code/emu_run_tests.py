import os.path
import time
import numpy as np
import scipy.stats as sps
from pyDOE import lhs
from inspect import getsourcefile
from os.path import abspath, dirname, basename, normpath
from pathlib import Path
from glob import glob
from testfunc_wrapper import TestFunc
from emu_single_test import single_test
from emu_run_plots import run_plots


def make_dirs():
    current_dir = abspath(dirname(getsourcefile(lambda: 0)))  # where this script is
    results_dir = r'\emulator_PCGPwM_results'
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
    nx = 15
    # Number of parameters
    ns = [250]

    # Knobs options
    fail_configs = [(True, 'low'), (True, 'high')] #, (False, 'low'), (False, 'high'), (None, 'none')]
    bigMs = [10]
    models = ['piston'] # 'borehole', 'otlcircuit', 'wingweight',
    emulator_methods = ['PCGPwM']# , 'GPy'] #, 'PCGP_KNN', 'PCGP_BR'] #

    # Query test function for Borehole
    func_caller = TestFunc(models[0]).info
    function_name = func_caller['function']
    xdim = func_caller['xdim']
    thetadim = func_caller['thetadim']

    # JSON filelist
    resultJSONs = []
    for j in js:
        np.random.seed(j)
        x = sps.uniform.rvs(0, 1, (nx, xdim))
        np.random.seed()

        testtheta = np.random.uniform(0, 1, (1000, thetadim))

        for n in ns:
            theta = lhs(thetadim, n)
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
                    for bigM in bigMs:
                        result_fname = \
                            single_test(method, x, theta, f, model, testtheta,
                                        function_name, n, fail_random, fail_level,
                                        bigM, j, data_dir, caller=func_caller)
                        resultJSONs.append(result_fname)
                        print(result_fname)
    return resultJSONs


if __name__ == '__main__':
    data_dir, plot_dir = make_dirs()

    start_time = time.time()
    listJSONs = run_experiment(data_dir)
    run_time = time.time() - start_time
    print('total runtime: {:.3f} seconds'.format(run_time))

    run_plots(plot_dir, listJSONs)
