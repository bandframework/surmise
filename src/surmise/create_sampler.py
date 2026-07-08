import copy
import warnings
import functools

from .utilitiesmethods.metropolis_hastings import sampler as sample_with_metropolis_hastings
from .utilitiesmethods.LMC import sampler as sample_with_LMC
from .utilitiesmethods.PTLMC import sampler as sample_with_PTLMC


def create_sampler(sampler_name, options):
    """
    Construct a sampler function for direct use by |surmise| calibrators.

    While this function is in the |surmise| public interface, for most use cases
    samplers are created under-the-hood automatically on behalf of the user.
    This is, therefore, an advanced feature made available to power users.

    Parameters
    ----------
    sampler_name : name of desired sampler offered by |surmise|
    options : ``dict`` of sampler-specific arguments that fully characterize the
        desired sampler.  Refer to the documentation of each sampler for more
        information.

    Returns
    -------
    The desired sampler function.  The following example demonstrates its use.

    .. code-block: python

        sample_with_PTLMC = surmise.create_sampler("PTLMC", ptlmc_args)
        results = sample_with_PTLMC(
            logpost_func=log_posterior,
            draw_func=draw_from_start_distribution,
            scipy_stats_rng=np.random.default_rng(RAND_SEED)
        )
    """
    KEY = "expertMode"

    if sampler_name.lower() == "metropolis_hastings":
        return functools.partial(sample_with_metropolis_hastings, **options)
    elif sampler_name.upper() == "LMC":
        lmc_options = copy.deepcopy(options)

        if KEY in lmc_options:
            if not isinstance(lmc_options[KEY], bool):
                raise ValueError(f"{KEY} value must be a boolean")
            elif not lmc_options[KEY]:
                msg = "{} is included for unofficial research purposes only"
                raise ValueError(msg.format(sampler_name))

            del lmc_options[KEY]
        else:
            msg = "{} is included for unofficial research purposes only"
            raise ValueError(msg.format(sampler_name))

        # Emit warning to extend a helping hand to the experts.
        msg = f"Using unofficial research {sampler_name} sampler"
        warnings.warn(msg)
        return functools.partial(sample_with_LMC, **lmc_options)
    elif sampler_name.upper() == "PTLMC":
        return functools.partial(sample_with_PTLMC, **options)

    raise TypeError(f"Invalid sampler ({sampler_name})")
