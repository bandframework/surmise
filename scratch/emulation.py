# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:12:24 2020

@author: Matt
"""

# import emulation_GPy
from emulation_default import emulation_builder_default, emulation_draws_default, \
    emulation_prediction_default


def emulation_builder(thetaval, fval, inputval, missingval = None, software = 'default', emuoptions = None):
    if software == 'default':
        emuinfo = emulation_builder_default(thetaval, fval, inputval, missingval, emuoptions)
        emuinfo['software'] = 'default'
    else:
        print ('Choose valid emulation software options. (\'default\' and \'GPy\')')
        raise

    return emuinfo

def emulation_draws(emuinfo, thetanew, inputnew = None, drawoptions = None, grid=None):
    if emuinfo['software'] == 'default':
        fdraws = emulation_draws_default(emuinfo, thetanew, drawoptions)
    else:
        print ('Choose valid emulation software options. (\'default\' and \'GPy\')')
        raise

    return fdraws

def emulation_prediction(emuinfo, thetanew, inputnew = None, grid=None):
    if emuinfo['software'] == 'default':
        yhat = emulation_prediction_default(emuinfo, thetanew)
    else:
        print ('Choose valid emulation software options. (\'default\' and \'GPy\')')
        raise

    return yhat
