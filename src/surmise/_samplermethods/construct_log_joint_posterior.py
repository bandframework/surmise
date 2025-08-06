import functools

import numpy as np


def log_joint_posterior(theta, log_joint_prior, log_likelihood):
    """
    Compute the log of theta's joint posterior PDF (up to a constant) using only
    the given log of theta's joint prior and the given log of the likelihood.

    Both log functions should return a value of -numpy.inf where the underlying
    function evaluates to zero.

    :param theta: 
    :param log_joint_prior:
    :param log_likelihood:
    """
    logpost = log_joint_prior(theta)
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
        # obtain the log-likelihood and the gradient of it
        loglikinds, dloglikinds = log_likelihood_grad(theta[inds, :])

        logpost[inds] += loglikinds
        dlogpost[inds] += dloglikinds

    return logpost, dlogpost


def construct_log_joint_posterior(thetaprior, log_likelihood, use_grad):
    """
    .. todo::
        * Should we perform checks of the two given functions as is typically
          done in surmise?
    """
    # Do not assume that the availability of the gradient means that we should
    # use it.
    if use_grad:
        if "lpdf_grad" not in dir(thetaprior):
            raise ValueError("Gradient of log joint prior not provided")

        # Assume that if they want to use gradients and they gave us the
        # gradient of the prior, then they have given us a likelihood function
        # that also returns the gradient.
        return functools.partial(
            log_joint_posterior_grad,
            log_joint_prior=thetaprior.lpdf,
            grad_log_joint_prior=thetaprior.lpdf_grad,
            log_likelihood=log_likelihood
        )
   
    return functools.partial(
        log_joint_posterior,
        log_joint_prior=thetaprior.lpdf,
        log_likelihood=log_likelihood
    )
