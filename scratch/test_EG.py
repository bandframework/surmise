import numpy as np
from surmise.emulation import emulator
import matplotlib.pyplot as plt


if __name__ == "__main__":
    paramval = np.loadtxt('testingdata/thetavals.csv',delimiter=',')
    fval = np.loadtxt('testingdata/functionevals.csv',delimiter=',')
    inputval = np.loadtxt('testingdata/inputdata.csv',delimiter=',',dtype='object')
    paramtrain = paramval[2920:3150,:]
    ftrain = fval[2920:3150,0:30]
    inputtrain = inputval[0:30]

    paramtest = paramval[1500:2749, :]
    ftest = fval[1500:2749, 0:30]

    missingmat = np.full_like(ftrain, False).astype(bool)
    missingmat[0:100, 0:15] = (np.random.rand(100, 15) > 0.75)

    ftrainmis = ftrain.copy()
    ftrainmis[missingmat] = np.nan

    emu = emulator(f=ftrainmis, theta=paramtrain, x=inputtrain,
                   method='GPEmGibbs',
                   args={'misval': missingmat, 'cat': True})

    pred = emu.predict()

    ftrainmisfill = ftrain.copy()
    ftrainmisfill[missingmat] = ftrain.mean()
    print(np.nanmean((ftrain - ftrainmisfill) ** 2))
    print(np.nanmean((pred.mean() - ftrainmis) ** 2))

    ftestmis = ftest.copy()

    missingmattest = np.full_like(ftest, False).astype(bool)
    missingmattest[0:100, 0:15] = (np.random.rand(100, 15) > 0.75)
    ftestmis[missingmattest] = np.nan
    testpred = emu.predict(x=inputtrain, theta=paramtest)

    print(np.nanmean((testpred.mean() - ftestmis) ** 2))
    print(testpred.var())
