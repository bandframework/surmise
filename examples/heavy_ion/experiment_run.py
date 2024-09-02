#!/usr/bin/env python3

from kfold_train_test.kfold_train_save_emu import kfold_train_save_emulators as kfold_train
from kfold_train_test.kfold_test_emu import kfold_test_emulators as kfold_test

import logging

# Ensure previous logging handlers are removed
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('exp_run.log', mode='w'),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

#-----------------------------------------------------

sim = 'Pb_Pb_2760_Grad'
emus = ['PCGP', 'PCGP_scikit', 'PCSK', 'AKSGP']
kfold_seeds_file = 'kfold_seeds.txt'
subdirs = 'kfold_set'


path_kfold_dir = kfold_train(simulation_name=sim, 
                             methods=emus, 
                             kfold_seeds_filename=kfold_seeds_file,
                             subdir_name=subdirs, 
                             regenerate=True, 
                             retrain=False)

kfold_test(path_kfold_data=path_kfold_dir,
           kfold_seeds_filename=kfold_seeds_file,
           methods=emus)



# import numpy as np
# from AKSGP import Emulator
# from time import time
# from surmise.emulation import emulator
# from metrics import rmse, normalized_rmse, intervalstats, wasserstein_distance_gaussian
# import pandas as pd


# def train_emu(emumethod, x, theta, f):
#     t0 = time()
#     if emumethod not in ['AKSGP', 'LCGP']:
#         emu = emulator(x=x, theta=theta, f=f, method=emumethod)
#     elif emumethod == 'AKSGP':
#         emu = Emulator(...)  # or load dill
#     elif emumethod == 'LCGP':
#         emu = None
#     t1 = time()
#     train_time = t1 - t0
#     return emu, train_time


# def predict_emu(emumethod, emu, x, theta):
#     predmean, predvar = None, None
#     t0 = time()
#     if emumethod not in ['AKSGP', 'LCGP']:
#         emupred = emu.predict(x=x, theta=theta)
#         predmean = emupred.mean()
#         predvar = emupred.var()
#     elif emumethod == 'AKSGP':
#         predmean, predvar = emu.predict( ... )
#     elif emumethod == 'LCGP':
#         pass
#     t1 = time()
#     predict_time = t1 - t0

#     return predmean, predvar, predict_time


# def read_data(dataset_name, run_id):
#     data_dict = ...
#     return data_dict

# def test_run(emumethod,
#              dataset_name,
#              run_id):
#     result = {
#         'run_id': run_id,
#         'dataset': dataset_name,
#         'emumethod': emumethod,
#         'rmse': None,
#         'nrmse': None,
#         'coverage': None,
#         'wasserstein_distance': None
#     }

#     data_dict = read_data(dataset_name, run_id)

#     (x, theta_train, f_train, theta_test, f_test) = data_dict[run_id]

#     emu, train_time = train_emu(emumethod, x=x, theta=theta_train, f=f_train)

#     predmean, predvar, predict_time = predict_emu(emumethod, emu=emu, x=x, theta=theta_test)

#     result['rmse'] = rmse(y=f_test, ypredmean=predmean)
#     result['coverage'], _ = intervalstats(y=f_test, ypredmean=predmean, ypredvar=predvar)
#     result['wasserstein_distance'] = wasserstein_distance_gaussian(...)
#     result['train_time'] = train_time
#     result['predict_time'] = predict_time

#     df = pd.DataFrame.from_dict(result)
#     df.to_csv('{:s}_{:s}_{:d}.csv'.format(emumethod, dataset_name, run_id))

#     return
