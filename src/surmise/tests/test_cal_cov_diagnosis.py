import warnings

import numpy as np
import pytest

from surmise.calibration import calibrator
from surmise.calibrationmethods._cov_diagnosis import (new_cov_diagnosis,
                                                       check_eigvals)
from . import shared_scenario as sc

pytestmark = pytest.mark.usefixtures('seeded_rng')

METHODS = ['directbayes', 'directbayeswoodbury']


@pytest.fixture
def flaky_eigh(monkeypatch):
    """Every 7th eigh call reports a negative smallest eigenvalue."""
    real_eigh = np.linalg.eigh
    calls = {'n': 0}

    def _eigh(a, *args, **kwargs):
        w, v = real_eigh(a, *args, **kwargs)
        calls['n'] += 1
        if calls['n'] % 7 == 0:
            w = w.copy()
            w[0] = -1e-3
        return w, v

    monkeypatch.setattr(np.linalg, 'eigh', _eigh)
    return calls


def _check_record(cov_diagnosis):
    n_rejected = cov_diagnosis['n_nonpd'] + cov_diagnosis['n_nonfinite']
    assert cov_diagnosis['n_nonpd'] > 0
    assert n_rejected <= cov_diagnosis['n_eval']
    assert cov_diagnosis['min_eig'] == -1e-3
    assert len(cov_diagnosis['theta']) == min(n_rejected,
                                              cov_diagnosis['max_store'])


@pytest.mark.parametrize('method', METHODS)
def test_nonpd_rejected_mh(method, emu_timedrop, flaky_eigh):
    with pytest.warns(RuntimeWarning, match='non-positive-definite'):
        cal = calibrator(emu=emu_timedrop, y=sc.y_td, x=sc.x_std,
                         thetaprior=sc.prior_balldrop, method=method,
                         yvar=sc.obsvar_td, args=sc.DEFAULT_MH_SPECS)
    _check_record(cal.info['cov_diagnosis'])
    assert np.all(np.isfinite(cal.theta.rnd(10)))


def test_nonpd_rejected_woodbury_grad(emu_lin_pcgpwm_wgrad, flaky_eigh):
    args = dict(sc.DEFAULT_PTLMC_SPECS)
    with pytest.warns(RuntimeWarning, match='non-positive-definite'):
        cal = calibrator(emu=emu_lin_pcgpwm_wgrad, y=sc.y_lin, x=sc.x_lin,
                         thetaprior=sc.priorphys_lin,
                         method='directbayeswoodbury',
                         yvar=sc.obsvar_lin, args=args)
    _check_record(cal.info['cov_diagnosis'])
    assert np.all(np.isfinite(cal.theta.rnd(10)))


@pytest.mark.parametrize('method', METHODS)
def test_clean_fit_records_no_problem(method, emu_timedrop):
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        cal = calibrator(emu=emu_timedrop, y=sc.y_td, x=sc.x_std,
                         thetaprior=sc.prior_balldrop, method=method,
                         yvar=sc.obsvar_td, args=sc.DEFAULT_MH_SPECS)
    cov_diagnosis = cal.info['cov_diagnosis']
    assert cov_diagnosis['n_eval'] > 0
    assert cov_diagnosis['n_nonpd'] == 0
    assert cov_diagnosis['n_nonfinite'] == 0


def test_check_eigvals_categories():
    cov_diagnosis = new_cov_diagnosis(max_store=2)
    theta = np.array([0.5])
    assert check_eigvals(cov_diagnosis, theta, np.array([1.0, 2.0]), 1.0)
    assert check_eigvals(cov_diagnosis, theta, np.array([0.5, 2.0]), 1.0)
    assert not check_eigvals(cov_diagnosis, theta, np.array([-1e-3, 2.0]), 1.0)
    assert not check_eigvals(cov_diagnosis, theta, np.array([1.0, 2.0]), 1.0,
                             arrays=(np.array([np.nan]),))
    assert not check_eigvals(cov_diagnosis, theta, np.array([np.nan, 1.0]),
                             1.0)
    assert cov_diagnosis['n_eval'] == 5
    assert cov_diagnosis['n_below_bound'] == 1
    assert cov_diagnosis['n_nonpd'] == 1
    assert cov_diagnosis['n_nonfinite'] == 2
    assert len(cov_diagnosis['theta']) == 2
