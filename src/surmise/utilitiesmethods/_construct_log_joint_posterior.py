import functools

import numpy as np


def log_joint_posterior(theta, log_joint_prior, log_likelihood):
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
    # TODO: Figure out final rules for type arguments of all these functions.
    # The fact that this module constructs the log joint posteriors, which are
    # passed to samplers in the private interface, should allow us to constrain
    # these choices strongly.
    logpost = np.atleast_1d(np.squeeze(log_joint_prior(theta)))
    inds = np.where(np.isfinite(logpost))[0]
    if len(inds) > 0:
        logpost[inds] += log_likelihood(theta[inds, :])

    return logpost


def log_joint_posterior_grad(theta, 
                             log_joint_prior, grad_log_joint_prior,
                             log_likelihood_grad):
    """
    Compute the log of theta's joint posterior PDF (up to a constant) using only
    the log of theta's joint prior and the given log of the likelihood as well
    as the gradients of these two functions.

    :param theta:
    :param log_joint_prior:
    :param grad_log_joint_prior:
    :param log_likelihood_grad:
    """
    logpost = log_joint_prior(theta)
    dlogpost = grad_log_joint_prior(theta)
    inds = np.where(np.isfinite(logpost))[0]

    if len(inds) > 0:
        loglikinds, dloglikinds = log_likelihood_grad(theta[inds, :])

        logpost[inds] += loglikinds
        dlogpost[inds] += dloglikinds

    return logpost, dlogpost


def construct_log_joint_posterior(thetaprior,
                                  log_likelihood, log_likelihood_grad):
    """
    .. todo::
        * Should we perform checks of the two given functions as was done
          previously in the surmise samplers?
    """
    logpost_func = functools.partial(
        log_joint_posterior,
        log_joint_prior=thetaprior.lpdf,
        log_likelihood=log_likelihood
    )

    # TODO: Check thetaprior as well?
    logpost_grad_func = None
    if callable(log_likelihood_grad):
        logpost_grad_func = functools.partial(
            log_joint_posterior_grad,
            log_joint_prior=thetaprior.lpdf,
            grad_log_joint_prior=thetaprior.lpdf_grad,
            log_likelihood=log_likelihood,
            log_likelihood_grad=log_likelihood_grad
        )

    return logpost_func, logpost_grad_func
