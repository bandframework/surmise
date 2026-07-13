"""Shared fixtures for src/surmise/tests.

Variables are fixed during collection, defined in scenarios.py;
fixtures here are built once instead of at import time
in every test.
"""

from contextlib import contextmanager

import numpy as np
import pytest

from surmise.emulation import emulator
from surmise.calibration import calibrator

from . import shared_scenario as sc
from .._RandomNumberGenerator import RandomNumberGenerator


# RNG helpers
@pytest.fixture
def no_rng():
    """For tests asserting the must-set-first error."""
    RandomNumberGenerator()._clear_RNG()
    yield


@contextmanager
def does_not_raise():
    """For parametrized expectations."""
    yield


@pytest.fixture(scope="session")
def lin_data():
    """linear model data"""
    return (sc.x_lin, sc.xv_lin, sc.theta_lin, sc.f_lin,
            sc.y_lin, sc.obsvar_lin)


@pytest.fixture(scope="session")
def emu_lin_pcgp():
    """PCGP emulator fit to the linear scenario."""
    return emulator(x=sc.x_lin, theta=sc.theta_lin, f=sc.f_lin,
                    method='PCGP')


@pytest.fixture(scope="session")
def emu_lin_pcgpwm():
    return emulator(x=sc.x_lin, theta=sc.theta_lin, f=sc.f_lin,
                    method='PCGPwM')


@pytest.fixture(scope="session")
def timedrop_data():
    """(x_std, theta, f, y, obsvar) for the timedrop scenario."""
    theta = sc.prior_balldrop.rnd(50)
    f = sc.timedrop(sc.x_std, theta, sc.x_range, sc.theta_range)
    return sc.x_std, theta, f, sc.y_td, sc.obsvar_td


@pytest.fixture(scope="session")
def emu_timedrop(timedrop_data):
    x_std, theta, f, _, _ = timedrop_data
    return emulator(x=x_std, theta=theta, f=f, method='PCGP')


@pytest.fixture(scope="session")
def cal_directbayes(emu_timedrop, timedrop_data):
    """Fitted directbayes calibrator."""
    x_std, _, _, y, obsvar = timedrop_data
    return calibrator(emu=emu_timedrop, y=y, x=x_std,
                      thetaprior=sc.prior_balldrop,
                      method='directbayes',
                      yvar=obsvar)