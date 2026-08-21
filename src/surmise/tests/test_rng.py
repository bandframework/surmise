import numpy as np
import pytest

from surmise import set_RNG
from surmise.emulation import emulator
from surmise.calibration import calibrator

from . import shared_scenario as sc


def test_emulator_requires_rng(no_rng):
    with pytest.raises(RuntimeError, match="set_RNG"):
        emulator(x=sc.x_lin, theta=sc.theta_lin, f=sc.f_lin,
                 method='PCGP')


def test_calibrator_requires_rng(emu_lin_pcgp, no_rng):
    # emu_lin_pcgp is built under _session_rng (session scope, so it is
    # constructed before function-scoped no_rng clears the singleton).
    with pytest.raises(RuntimeError, match="set_RNG"):
        calibrator(emu=emu_lin_pcgp,
                   y=sc.y_lin,
                   x=sc.x_lin,
                   thetaprior=sc.priorphys_lin,
                   method='directbayes',
                   yvar=sc.obsvar_lin,
                   args=sc.DEFAULT_MH_SPECS
                   )


def test_calibrator_methods_raise_after_clear(cal_directbayes, no_rng):
    """A fitted calibrator must not have stored an RNG reference."""
    with pytest.raises(RuntimeError, match="set_RNG"):
        cal_directbayes.theta.rnd(10)


# Test to reproduce results
def _build_and_draw(seed):
    """Seed the whole sequence, fit emu + cal, return posterior draws."""
    set_RNG(np.random.default_rng(seed))
    emu = emulator(x=sc.x_lin, theta=sc.theta_lin, f=sc.f_lin,
                   method='PCGP')
    cal = calibrator(emu=emu,
                     y=sc.y_lin,
                     x=sc.x_lin,
                     thetaprior=sc.priorphys_lin,
                     method='directbayes',
                     yvar=sc.obsvar_lin,
                     args=sc.DEFAULT_MH_SPECS)
    return cal.theta.rnd(10)


def test_emu_cal_reproducible():
    # same RNGs should return the same samples
    draws1 = _build_and_draw(123)
    draws2 = _build_and_draw(123)
    assert np.array_equal(draws1, draws2, equal_nan=False)


def test_emu_cal_seed_sensitivity():
    # different RNGs should return different samples
    draws1 = _build_and_draw(123)
    draws2 = _build_and_draw(456)
    assert draws1.shape == draws2.shape
    assert not np.array_equal(draws1, draws2, equal_nan=False)
