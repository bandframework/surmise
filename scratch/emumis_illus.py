# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:47:50 2020

@author: mosesyhc
"""
from emulation import *

import numpy as np

def emulation_test_multi():
    paramval = np.loadtxt('testingdata/thetavals.csv',delimiter=',')
    fval = np.loadtxt('testingdata/functionevals.csv',delimiter=',')
    inputval = np.loadtxt('testingdata/inputdata.csv',delimiter=',',dtype='object')
    paramtrain = paramval[2920:3150,:]
    ftrain = fval[2920:3150,0:30]
    inputtrain = inputval[0:30]
    
    paramtest = paramval[1500:2749, :]
    ftest = fval[1500:2749, 0:30]
    
    missingmat = np.zeros(ftrain.shape)
    missingmat[0:100,0:15] = (np.random.rand(100,15) > 0.25)
    missingmat[0, 0] = 1
    missingmat[0, 1] = 0
    missingmat[1, 0] = 0
    missingmat[1, 1] = 1
    ftrainmiss = np.ones(ftrain.shape)
    ftrainmiss = 0+1*ftrain[:]

    for k in range(0,ftrain.shape[1]):
        for l in range(0,ftrain.shape[0]):
            if (missingmat[l, k] > 0.5):
                ftrainmiss[l, k] = 0
                
    print(np.mean( (ftrainmiss - ftrain) ** 2))


    model =emulation_builder(paramtrain, ftrainmiss, inputtrain, missingmat,
                             emuoptions = {'blocking': 'individual',
                                           'modeltype': 'parasep',
                                           'corrthetafname': 'matern',
                                           'corrxfname': 'exp',
                                           'em_gibbs': True})
    print('model5')
    prednow = emulation_prediction(model, paramtest)
    drawsnow = emulation_draws(model, paramtest)
    print(np.mean( (prednow - ftest) ** 2))
    print(np.mean( (np.mean(drawsnow,2) - ftest) ** 2))
    print(np.mean( ((np.mean(drawsnow,2) - ftest) ** 2)/np.var(drawsnow,2)))

    #
    # model = emulation_builder(paramtrain, ftrain, inputtrain)
    # print('model0')
    # prednow = emulation_prediction(model, paramtest)
    # drawsnow = emulation_draws(model, paramtest, {'rowmarginals': True})
    # print(np.mean( (prednow - ftest) ** 2))
    # print(np.mean( (np.mean(drawsnow,2) - ftest) ** 2))
    # print(np.mean( ((np.mean(drawsnow,2) - ftest) ** 2)/np.var(drawsnow,2)))
    # normresid = ((np.mean(drawsnow,2) - ftest))/np.sqrt(np.var(drawsnow,2))
    # _ = plt.hist(np.matrix.flatten(normresid[0:500,:]), bins='auto')
    # plt.show()
    
    
    
    # model = emulation_builder(paramtrain, ftrain, inputtrain, 0*missingmat)
    # print('model1')
    # prednow = emulation_prediction(model, paramtest)
    # drawsnow = emulation_draws(model, paramtest)
    # print(np.mean( (prednow - ftest) ** 2))
    # print(np.mean( (np.mean(drawsnow,2) - ftest) ** 2))
    # print(np.mean( ((np.mean(drawsnow,2) - ftest) ** 2)/np.var(drawsnow,2)))
    
    #
    # model = emulation_builder(paramtrain, ftrainmiss, inputtrain, missingmat, emuoptions = {'corrthetafname': 'exp', 'modeltype': 'parasep'})
    # print('model3')
    # prednow = emulation_prediction(model, paramtest)
    # drawsnow = emulation_draws(model, paramtest)
    # print(np.mean( (prednow - ftest) ** 2))
    # print(np.mean( (np.mean(drawsnow,2) - ftest) ** 2))
    # print(np.mean( ((np.mean(drawsnow,2) - ftest) ** 2)/np.var(drawsnow,2)))
    #
    # model = emulation_builder(paramtrain, ftrainmiss, inputtrain, missingmat)
    # print('model2')
    # prednow = emulation_prediction(model, paramtest)
    # drawsnow = emulation_draws(model, paramtest)
    # print(np.mean( (prednow - ftest) ** 2))
    # print(np.mean( (np.mean(drawsnow,2) - ftest) ** 2))
    # print(np.mean( ((np.mean(drawsnow,2) - ftest) ** 2)/np.var(drawsnow,2)))
    #
    # model = emulation_builder(paramtrain, ftrainmiss, inputtrain, missingmat, emuoptions = {'corrthetafname': 'exp', 'modeltype': 'parasep'})
    # print('model3')
    # prednow = emulation_prediction(model, paramtest)
    # drawsnow = emulation_draws(model, paramtest)
    # print(np.mean( (prednow - ftest) ** 2))
    # print(np.mean( (np.mean(drawsnow,2) - ftest) ** 2))
    # print(np.mean( ((np.mean(drawsnow,2) - ftest) ** 2)/np.var(drawsnow,2)))
    #
    # model = emulation_builder(paramtrain, ftrainmiss, inputtrain, missingmat, emuoptions = {'corrxfname': 'matern', 'modeltype': 'parasep'})
    # print('model3')
    # prednow = emulation_prediction(model, paramtest)
    # drawsnow = emulation_draws(model, paramtest)
    # print(np.mean( (prednow - ftest) ** 2))
    # print(np.mean( (np.mean(drawsnow,2) - ftest) ** 2))
    # print(np.mean( ((np.mean(drawsnow,2) - ftest) ** 2)/np.var(drawsnow,2)))

    return

if __name__ == "__main__":
    emulation_test_multi()