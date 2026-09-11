import functools

import numpy as np


def log_joint_posterior(theta, return_grad, log_joint_prior, log_likelihood):
    """
    Compute the log of theta's joint posterior PDF (up to a constant) using only
    the given log of theta's joint prior and the given log of the likelihood.

    Both log functions should return a value of -numpy.inf where the underlying
    function evaluates to zero.

    :param theta:
    :param return_grad:
    :param log_joint_prior:
    :param log_likelihood:
    """
    assert not return_grad

    logpost = log_joint_prior(theta)
    inds = np.where(np.isfinite(logpost))[0]
    if len(inds) > 0:
        logpost[inds] += log_likelihood(theta[inds, :])

    return logpost


def construct_log_joint_posterior(thetaprior, log_likelihood, has_grad):
    """
    .. todo::
        * Should we perform checks of the two given functions as is typically
          done in surmise?
    """
    assert not has_grad

    return functools.partial(
        log_joint_posterior,
        log_joint_prior=thetaprior.lpdf,
        log_likelihood=log_likelihood
    )
