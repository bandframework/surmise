"""
Bookkeeping for likelihood evaluations whose covariance is numerically unusable.

A calibration method creates a record with :func:`new_cov_diagnosis` before
sampling, calls :func:`check_eigvals` once per theta inside its likelihood, and
calls :func:`warn_cov_diagnosis` after sampling.  Rejected theta receive a log
likelihood of ``-inf`` so that samplers treat them like zero prior density.
"""
import warnings

import numpy as np


def new_cov_diagnosis(max_store=50):
    """Return an empty record of covariance problems."""
    return {'n_eval': 0,           # likelihood evaluations checked
            'n_nonfinite': 0,      # rejected: NaN/inf in mean, cov or eigvals
            'n_nonpd': 0,          # rejected: smallest eigenvalue <= tol
            'n_below_bound': 0,    # kept, but min eigenvalue < analytic bound
            'min_eig': np.inf,
            'theta': [], 'eig': [], 'max_store': max_store}


def check_eigvals(cov_diagnosis, theta, eigvals, lower_bound, arrays=()):
    """
    Record one evaluation and return ``True`` if the likelihood is usable.

    Parameters
    ----------
    cov_diagnosis : dict
        Record created by :func:`new_cov_diagnosis`.
    theta : numpy.ndarray
        Parameter value being evaluated, stored for failed evaluations.
    eigvals : numpy.ndarray
        Eigenvalues of the matrix whose log-determinant and inverse are used.
    lower_bound : float
        Smallest eigenvalue possible in exact arithmetic, e.g. ``min(obsvar)``
        for ``C + diag(obsvar)`` or ``1`` for ``I + J^T J``.
    arrays : tuple of numpy.ndarray
        Other inputs (mean, covariance factors) that must be finite.
    """
    cov_diagnosis['n_eval'] += 1
    finite = all(np.all(np.isfinite(a)) for a in (*arrays, eigvals))
    if finite:
        mineig = np.min(eigvals)
        cov_diagnosis['min_eig'] = min(cov_diagnosis['min_eig'], mineig)
        tol = eigvals.size * np.finfo(float).eps * np.max(np.abs(eigvals))
        if mineig > tol:
            if mineig < lower_bound - tol:
                cov_diagnosis['n_below_bound'] += 1
            return True
        cov_diagnosis['n_nonpd'] += 1
    else:
        cov_diagnosis['n_nonfinite'] += 1

    if len(cov_diagnosis['theta']) < cov_diagnosis['max_store']:
        cov_diagnosis['theta'].append(np.array(theta, copy=True))
        cov_diagnosis['eig'].append(np.sort(np.ravel(eigvals))[:3].copy())
    return False


def warn_cov_diagnosis(cov_diagnosis, method):
    """Warn once if any evaluation was rejected or lost accuracy."""
    n_eval = cov_diagnosis['n_eval']
    n_rejected = cov_diagnosis['n_nonfinite'] + cov_diagnosis['n_nonpd']
    if n_rejected > 0:
        warnings.warn(
            f"{method}: {n_rejected} of {n_eval} likelihood evaluations were "
            f"rejected ({cov_diagnosis['n_nonpd']} non-positive-definite, "
            f"{cov_diagnosis['n_nonfinite']} non-finite; min eigenvalue "
            f"{cov_diagnosis['min_eig']:.3g}). "
            "See calibrator.info['cov_diagnosis'].", RuntimeWarning)
    elif cov_diagnosis['n_below_bound'] > 0:
        warnings.warn(
            f"{method}: {cov_diagnosis['n_below_bound']} of {n_eval} "
            "likelihood covariances had eigenvalues below their analytic "
            "lower bound; results may be inaccurate. "
            "See calibrator.info['cov_diagnosis'].", RuntimeWarning)
