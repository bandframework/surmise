import warnings

from .utilitiesmethods.metropolis_hastings import sampler as sample_with_metropolis_hastings
from .utilitiesmethods.LMC import sampler as sample_with_LMC
from .utilitiesmethods.PTLMC import sampler as sample_with_PTLMC


def create_sampler(sampler, expert_mode):
    """
    Construct a sampler function for direct use by |surmise| calibrators.  The
    following example demonstrates its use.

    .. code-block:: python

        sample_with_PTLMC = surmise.create_sampler("PTLMC", expert_mode=False)
        results = sample_with_PTLMC(
            logpost_func=log_posterior,
            draw_func=draw_from_start_distribution,
            scipy_stats_rng=np.random.default_rng(RAND_SEED),
            **pltlmc_args
        )

    For typical use cases, samplers are created automatically under-the-hood on
    behalf of users.  Therefore, there is generally no need to explicitly create
    or access samplers.  This function is in the |surmise| public interface only
    as an advanced feature for use by developers and power users.

    .. todo::
        * The samplers should be updated so that they accept a single dictionary
          containing all sampler arguments.  The samplers can then error check
          those hard and print useful error messages.

    Parameters
    ----------
    sampler :
        Name of desired sampler offered by |surmise|
    expert_mode :
        Allow the use of research-grade samplers if ``True``

    Returns
    -------
    :
        The desired sampler function.
    """
    if isinstance(sampler, str):
        if sampler.lower() == "metropolis_hastings":
            return sample_with_metropolis_hastings
        elif sampler.upper() == "LMC":
            if not expert_mode:
                msg = "{} is included for unofficial research purposes only"
                raise ValueError(msg.format(sampler))

            # Emit warning to extend a helping hand to the experts.
            msg = f"Using unofficial research {sampler} sampler"
            warnings.warn(msg)
            return sample_with_LMC
        elif sampler.upper() == "PTLMC":
            return sample_with_PTLMC
    elif isinstance(sampler, dict):
        if len(sampler) != 1:
            return ValueError('Custom sampler must be {"user": my_sampler_fcn}')
        source = sampler.keys()[0]
        if source.lower() != "user":
            return ValueError('Custom sampler must be {"user": my_sampler_fcn}')
        sampler_fcn = sampler[source]
        if not callable(sampler_fcn):
            return ValueError("Custom sampler function is not callable")

        raise NotImplementedError("This functionality is not under test")
    else:
        raise TypeError(f"Sampler should be a string or dict ({sampler})")

    raise ValueError(f"Invalid sampler ({sampler})")
