##############################################
#            Simple scenarios                #
##############################################
import numpy as np
import pytest
from surmise.calibration import calibrator
##############################################
#            Simple scenarios                #
##############################################

# ..note:
#  pytest collects all tests (and run portions of variable declarations) before deselecting tests.
#  As a result a .set_RNG routine elsewhere will propagate into other tests even if not intended.
#  The use of fixture `no_rng` is to enforce the error raised from an unset RNG

# from .conftest import no_rng, emu_lin_pcgp
from .shared_scenario import x_lin as x, y_lin as y, \
                             obsvar_lin as obsvar, priorphys_lin


##############################################
# Unit tests to initialize an emulator class #
##############################################
args = {'theta0': np.array([[0.4]]),
        'numsamp': 20,
        'stepType': 'normal',
        'stepParam': [0.4]}


def test_cal_rng_notset(no_rng, emu_lin_pcgp):
    with pytest.raises(RuntimeError):
        _ = calibrator(emu=emu_lin_pcgp,
                       y=y,
                       x=x,
                       thetaprior=priorphys_lin,
                       method='directbayes',
                       yvar=obsvar,
                       args=args)
