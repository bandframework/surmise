import numpy as np
from surmise.emulation import emulator
import pytest
import os

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f

pytestmark = pytest.mark.usefixtures('seeded_rng')


##############################################
#            Simple scenarios                #
##############################################
f1 = f[0:15, :]
f2 = f[:, 0:25]
theta1 = theta[0:25, :]
x1 = x[0:15, :]
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)

##############################################
# Unit tests to initialize an emulator class #
##############################################


@pytest.mark.parametrize(
    "load_emu_flag, expectation",
    [
     (True, does_not_raise()),
     (False, pytest.raises(TypeError))
     ],
    )
def test_emu_saveload(load_emu_flag, expectation):
    fname = 'test_emu_saveload.pkl'
    with expectation:
        emu = emulator(x=x, theta=theta, f=f)
        emu.save_to(fname)

        if load_emu_flag:
            emuload = emulator.load_from(fname)
        else:
            try:
                emuload = emulator.load_prediction(fname)
            except TypeError:
                # in case test fails, generated files should be cleaned up
                os.remove(fname)
                raise TypeError
        assert emuload is not None
        os.remove(fname)


def test_emupred_saveload():
    fname = 'test_emupred_saveload.pkl'
    with does_not_raise():
        emu = emulator(x=x, theta=theta, f=f)

        emupred = emu.predict()
        emupred.save_to(fname)

        emupredload = emulator.load_prediction(fname)
        assert (emupredload.mean() == emupred.mean()).all()

        os.remove(fname)
