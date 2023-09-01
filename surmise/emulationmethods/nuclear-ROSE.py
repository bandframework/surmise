"""nuclear-ROSE, an integration wrapper for using emulators produced by [fill-in] nuclear-rose package."""
import numpy as np


def fit(fitinfo, rose_emu, **kwargs):
    '''
    The purpose of fit is to take information and plug all of our fit
    information into fitinfo, which is a python dictionary.

    Example usage for ROSE:
    rbm = emulator(method='nuclear-ROSE',
            args={'rose_emu': <<insert rose.ScatteringAmplitudeEmulator object>>})

    Parameters
    ----------
    fitinfo : dict
        A dictionary including the emulation fitting information once
        complete.
        The dictionary is passed by reference, so it returns None.
    args : dict, optional
        A dictionary containing options. The default is None. Insert ROSE emulator with
        key = 'rose_emu'.

    Returns
    -------
    None.

    '''
    fitinfo['emulator'] = rose_emu

    return


def predict(predinfo, fitinfo, x, theta, **kwargs):
    '''
    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where you should place all of your prediction
        information once complete.

        - predinfo['mean'] : mean prediction.

        - predinfo['cov'] : variance of the prediction.

    x : numpy.ndarray
        An array of inputs. Each row should correspond to a row in f.

    theta : numpy.ndarray
        An array of parameters. Each row should correspond to a column in f.

    args : dict, optional
        A dictionary containing options. The default is None.

    Returns
    -------
    Prediction mean and variance at theta and x given the dictionary fitinfo.
    '''

    rose_emu = fitinfo['emulator']
    outputArray = []

    for i, thetai in enumerate(theta):
        amplitudeEm = rose_emu.emulate_dsdo(thetai)
        outputArray.append(amplitudeEm[x])

    predmean = np.array(outputArray)

    predinfo['mean'] = predmean
    predinfo['var'] = np.zeros_like(predmean)
    return
