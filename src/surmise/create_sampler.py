import warnings

from .utilitiesmethods.metropolis_hastings import sampler as sample_with_metropolis_hastings
from .utilitiesmethods.LMC import sampler as sample_with_LMC
from .utilitiesmethods.PTLMC import sampler as sample_with_PTLMC
from .utilitiesmethods._construct_log_joint_posterior import construct_log_joint_posterior


def create_sampler(description, expert_mode):
    """
    Construct a sampler function for direct use by |surmise| calibrators.  The
    following example demonstrates its use.

    .. code-block:: python

        sample_with_PTLMC = surmise.create_sampler("PTLMC", expert_mode=False)
        results = sample_with_PTLMC(
            logpost_func=log_posterior,
            draw_func=draw_from_start_distribution,
            scipy_stats_rng=np.random.default_rng(RAND_SEED),
            specification=ptlmc_spec
        )

    For typical use cases, samplers are created automatically under-the-hood on
    behalf of users.  Therefore, there is generally no need to explicitly create
    or access samplers.  This function is in the |surmise| public interface only
    as an advanced feature for use by developers and power users.

    Parameters
    ----------
    description :
        Name of desired sampler offered by |surmise|.  Valid values are

        * "metropolis_hastings" to use
          :py:func:`surmise.utilitiesmethods.sample_with_metropolis_hastings`
        * "LMC" to use **research-grade**
          :py:func:`surmise.utilitiesmethods.sample_with_LMC`
        * "PTLMC" to use
          :py:func:`surmise.utilitiesmethods.sample_with_PTLMC`

    expert_mode :
        Allow the use of a research-grade sampler if ``True``

    Returns
    -------
    :
        The desired sampler function.
    """
    # User-provided samplers must have an interface explicitly linked to
    # sampling posteriors -- they must accept log joint prior and log
    # likelihood.
    if isinstance(description, dict):
        if len(description) != 1:
            return ValueError('Custom sampler must be {"user": my_sampler_fcn}')
        source = list(description.keys())[0]
        if source.lower() != "user":
            return ValueError('Custom sampler must be {"user": my_sampler_fcn}')
        sampler_fcn = description[source]
        if not callable(sampler_fcn):
            return ValueError("Custom sampler function is not callable")

        return sampler_fcn
    elif not isinstance(description, str):
        raise TypeError(f"description should be a string or dict ({description})")

    # We wrap internal samplers so that these can remain as general use MCMC
    # samplers.
    if description.lower() == "metropolis_hastings":
        sampler = sample_with_metropolis_hastings
    elif description.upper() == "LMC":
        if not expert_mode:
            msg = "{} is included for unofficial research purposes only"
            raise ValueError(msg.format(description))

        # Emit warning to extend a helping hand to the experts.
        msg = f"Using unofficial research {description} sampler"
        warnings.warn(msg)
        sampler = sample_with_LMC
    elif description.upper() == "PTLMC":
        sampler = sample_with_PTLMC
    else:
        raise ValueError(f"Invalid sampler ({description})")

    def _sampler_wrapped(log_joint_prior, log_likelihood,
                         draw_func, scipy_stats_rng, specification):
        log_joint_posterior = construct_log_joint_posterior(
            log_joint_prior, log_likelihood, has_grad=False
        )
        return sampler(
            logpost_func=log_joint_posterior,
            draw_func=draw_func,
            scipy_stats_rng=scipy_stats_rng,
            specification=specification
        )

    return _sampler_wrapped
