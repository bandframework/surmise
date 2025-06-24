"""nuclear-ROSE, an integration wrapper for using emulators produced by [fill-in] nuclear-rose package."""
import numpy as np
import bisect


def fit(fitinfo, rose_emu, emu_variance_constant=0.0, angle_atol=1e-2, **kwargs):
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
    angle_atol : scalar, optional
        User-defined absolute tolerance between emulated angles and angles to predict at.
        Prediction at the closest emulated angle within the tolerance is currently returned.
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
    fitinfo['angle_atol'] = angle_atol

    return


def predict(predinfo, fitinfo, x: np.ndarray, theta: np.ndarray, **kwargs):
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
    atol = fitinfo['angle_atol']

    xinds = find_closest_angles(predict_angles=x, full_angles=rose_emu.angles, angle_atol=atol)

    outputArray = []
    for i in range(len(theta)):
        amplitudeEm = rose_emu.emulate_dsdo(theta[i])
        outputArray.append(amplitudeEm[xinds])

    predmean = np.array(outputArray).T

    predinfo['mean'] = predmean
    predinfo['var'] = np.ones_like(predmean) * fitinfo['emulator_variance_constant']
    return


def find_closest_angles(predict_angles: np.ndarray, full_angles: np.ndarray, angle_atol: float):
    """
    Finds the closest emulated angles and returns their indices.

    Parameters
    ----------
    predict_angles : np.ndarray
        The array of angles to predict at.
    full_angles : np.ndarray
        The array of angles that have been emulated at.
    angle_atol : scalar
        The absolute tolerance between any entry of predict_angles and its closest angle in full_angles.
        If the distance is larger than the absolute tolerance, raise an AssertionError.

    Returns
    -------
    xinds : list
        List of indices of emulated angles.

    """
    sortind = np.argsort(full_angles)
    sort_emu_angles = full_angles[sortind]
    xinds = []
    for i in range(len(predict_angles)):
        i1 = bisect.bisect_left(sort_emu_angles, predict_angles[i])
        i2 = bisect.bisect_right(sort_emu_angles, predict_angles[i])
        if predict_angles[i] - sort_emu_angles[i1] <= sort_emu_angles[i2] - predict_angles[i]:
            xinds.append(i1)
        else:
            xinds.append(i2)

    if (np.array(xinds) >= len(full_angles)).all() and ~np.allclose(predict_angles, full_angles[xinds]):
        raise ValueError('The angles to predict at are all larger than the emulated angles beyond the absolute '
                         'tolerance.')

    assert np.allclose(predict_angles, full_angles[xinds], atol=angle_atol), \
        ('requested angles should be close to one of the emulated angles `SAE.angles`,'
         'with an absolute tolerance of {:.2E}.'.format(angle_atol))
    return xinds
