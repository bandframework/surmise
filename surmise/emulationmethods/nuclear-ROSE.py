"""nuclear-ROSE, an integration wrapper for using emulators produced by [fill-in] nuclear-rose package."""
import numpy as np
import bisect


def fit(fitinfo, rose_emu, emu_variance_constant=0.0, **kwargs):
    '''
    The purpose of fit is to take information and plug all of our fit
    information into fitinfo, which is a python dictionary.

    Example usage for ROSE:
    rbm = emulator(method='nuclear-ROSE',
            args={'rose_emu': <<insert rose.ScatteringAmplitudeEmulator object>>,
                  'emu_variance_constant': 1e-5})

    Parameters
    ----------
    fitinfo : dict
        A dictionary including the emulation fitting information once complete.
        The dictionary is passed by reference, so it returns None.
    rose_emu : nuclear-rose.ScatteringAmplitudeEmulator
        ROSE emulator object.
    emu_variance_constant : scalar, optional
        User-defined emulator variance constant.  For example, one may estimate
        this quantity empirically.  The default is 0.
    args : dict, optional
        A dictionary containing options. The default is None. Insert ROSE emulator with
        key = 'rose_emu'.

    Returns
    -------
    None.

    '''
    assert emu_variance_constant >= 0, 'Emulator variance must be nonnegative.'

    fitinfo['emulator'] = rose_emu
    fitinfo['emulator_variance_constant'] = emu_variance_constant

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
    emu_angles = rose_emu.angles

    sortind = np.argsort(emu_angles)
    sort_emu_angles = emu_angles[sortind]
    xinds = []
    for i in range(len(x)):
        i1 = bisect.bisect_left(sort_emu_angles, x[i])
        i2 = bisect.bisect_right(sort_emu_angles, x[i])
        if i1 == i2:
            xinds.append(i1)
        else:
            xinds.append(sortind[i1:i2][0])

    assert np.allclose(x, emu_angles[xinds]), 'requested angles should be a subset of emulated angles `SAE.angles`.'

    outputArray = []
    for i in range(len(theta)):
        amplitudeEm = rose_emu.emulate_dsdo(theta[i])
        outputArray.append(amplitudeEm[xinds])

    predmean = np.array(outputArray).T

    predinfo['mean'] = predmean
    predinfo['var'] = np.ones_like(predmean) * fitinfo['emulator_variance_constant']
    return
