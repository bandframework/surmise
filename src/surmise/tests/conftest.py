"""Shared fixtures for src/surmise/tests.

Variables are fixed during collection, defined in scenarios.py;
fixtures here are built once instead of at import time
in every test.
"""
import numpy as np
from contextlib import contextmanager

import pytest

from surmise.emulation import emulator
from surmise.calibration import calibrator

from . import shared_scenario as sc
from .._RandomNumberGenerator import RandomNumberGenerator
from surmise import set_RNG


@pytest.fixture(scope="session")
def _session_rng():
    def _ensure():
        set_RNG(np.random.default_rng(sc.RNG_SEED))
    _ensure()
    return _ensure


@pytest.fixture(autouse=True)
def _isolate_rng_state():
    singleton = RandomNumberGenerator()
    try:
        saved = singleton.scipy_stats_RNG
    except RuntimeError:
        saved = None
    yield
    if saved is None:
        singleton._clear_RNG()
    else:
        singleton.scipy_stats_RNG = saved


# RNG helpers
@pytest.fixture
def no_rng():
    """For tests asserting the must-set-first error."""
    RandomNumberGenerator()._clear_RNG()
    yield


@pytest.fixture(scope="module")
def seeded_rng():
    """Set the package RNG once for an entire test module."""
    _rng = np.random.default_rng(sc.RNG_SEED)
    set_RNG(_rng)
    yield _rng
    RandomNumberGenerator()._clear_RNG()


@contextmanager
def does_not_raise():
    """For parametrized expectations."""
    yield


@pytest.fixture(scope="session")
def lin_data(_session_rng):
    """linear model data"""
    return (sc.x_lin, sc.xv_lin, sc.theta_lin, sc.f_lin,
            sc.y_lin, sc.obsvar_lin)


@pytest.fixture(scope="session")
def emu_lin_pcgp(_session_rng):
    """PCGP emulator fit to the linear scenario."""
    _session_rng()
    return emulator(x=sc.x_lin, theta=sc.theta_lin, f=sc.f_lin,
                    method='PCGP')


@pytest.fixture(scope="session")
def emu_lin_pcgpwm(_session_rng):
    _session_rng()
    return emulator(x=sc.x_lin, theta=sc.theta_lin, f=sc.f_lin,
                    method='PCGPwM')


@pytest.fixture(scope="session")
def emu_lin_pcgpwm_wgrad(_session_rng):
    _session_rng()
    return emulator(x=sc.x_lin, theta=sc.theta_lin, f=sc.f_lin,
                    method='PCGPwM',
                    args={'return_grad': True})


@pytest.fixture(scope="session")
def timedrop_data(_session_rng):
    """(x_std, theta, f, y, obsvar) for the timedrop scenario."""
    theta = sc.prior_balldrop.rnd(50)
    f = sc.timedrop(sc.x_std, theta, sc.x_range, sc.theta_range)
    return sc.x_std, theta, f, sc.y_td, sc.obsvar_td


@pytest.fixture(scope="session")
def emu_timedrop(_session_rng, timedrop_data):
    _session_rng()
    x_std, theta, f, _, _ = timedrop_data
    return emulator(x=x_std, theta=theta, f=f, method='PCGP')


@pytest.fixture(scope="session")
def cal_directbayes(_session_rng, emu_timedrop, timedrop_data):
    """Fitted directbayes calibrator."""
    _session_rng()
    x_std, _, _, y, obsvar = timedrop_data
    return calibrator(emu=emu_timedrop, y=y, x=x_std,
                      thetaprior=sc.prior_balldrop,
                      method='directbayes',
                      yvar=obsvar,
                      args=sc.DEFAULT_MH_SPECS)
