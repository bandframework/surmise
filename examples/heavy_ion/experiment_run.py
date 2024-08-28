import numpy as np
from AKSGP import Emulator
from time import time
from surmise.emulation import emulator
from metrics import rmse, normalized_rmse, intervalstats, wasserstein_distance_gaussian
import pandas as pd


def train_emu(emumethod, x, theta, f):
    t0 = time()
    if emumethod not in ['AKSGP', 'LCGP']:
        emu = emulator(x=x, theta=theta, f=f, method=emumethod)
    elif emumethod == 'AKSGP':
        emu = Emulator(...)  # or load dill
    elif emumethod == 'LCGP':
        emu = None
    t1 = time()
    train_time = t1 - t0
    return emu, train_time


def predict_emu(emumethod, emu, x, theta):
    predmean, predvar = None, None
    t0 = time()
    if emumethod not in ['AKSGP', 'LCGP']:
        emupred = emu.predict(x=x, theta=theta)
        predmean = emupred.mean()
        predvar = emupred.var()
    elif emumethod == 'AKSGP':
        predmean, predvar = emu.predict( ... )
    elif emumethod == 'LCGP':
        pass
    t1 = time()
    predict_time = t1 - t0

    return predmean, predvar, predict_time


def read_data(dataset_name, run_id):
    data_dict = ...
    return data_dict

def test_run(emumethod,
             dataset_name,
             run_id):
    result = {
        'run_id': run_id,
        'dataset': dataset_name,
        'emumethod': emumethod,
        'rmse': None,
        'nrmse': None,
        'coverage': None,
        'wasserstein_distance': None
    }

    data_dict = read_data(dataset_name, run_id)

    (x, theta_train, f_train, theta_test, f_test) = data_dict[run_id]

    emu, train_time = train_emu(emumethod, x=x, theta=theta_train, f=f_train)

    predmean, predvar, predict_time = predict_emu(emumethod, emu=emu, x=x, theta=theta_test)

    result['rmse'] = rmse(y=f_test, ypredmean=predmean)
    result['coverage'], _ = intervalstats(y=f_test, ypredmean=predmean, ypredvar=predvar)
    result['wasserstein_distance'] = wasserstein_distance_gaussian(...)
    result['train_time'] = train_time
    result['predict_time'] = predict_time

    df = pd.DataFrame.from_dict(result)
    df.to_csv('{:s}_{:s}_{:d}.csv'.format(emumethod, dataset_name, run_id))

    return
