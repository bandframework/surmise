import copy
import warnings
import functools

from .utilitiesmethods.metropolis_hastings import sampler as sample_with_metropolis_hastings
from .utilitiesmethods.LMC import sampler as sample_with_LMC
from .utilitiesmethods.PTLMC import sampler as sample_with_PTLMC


def create_sampler(sampler, options):
    """
    Construct a sampler function for direct use by |surmise| calibrators.  The
    following example demonstrates its use.

    .. code-block:: python

        sample_with_PTLMC = surmise.create_sampler("PTLMC", ptlmc_args)
        results = sample_with_PTLMC(
            logpost_func=log_posterior,
            draw_func=draw_from_start_distribution,
            scipy_stats_rng=np.random.default_rng(RAND_SEED)
        )

    For typical use cases, samplers are created automatically under-the-hood on
    behalf of users.  Therefore, there is generally no need to explicitly create
    or access samplers.  This function is in the |surmise| public interface only
    as an advanced feature for use by developers and power users.

    .. todo::
        * The current implementation prevents users from providing a custom
          sampler to calibrators.  Consider allowing ``sampler`` to be a
          user-provided sampler function that we assume has the necessary
          interface.  This function could then confirm that options is ``None``
          or an empty ``dict`` and just pass that function along.  If the
          calibrator interface is updated so that calling code must provide a
          sampler identifier, then that argument could also be setup in this
          same way.

    Parameters
    ----------
    sampler :
        Name of desired sampler offered by |surmise|
    options :
        ``dict`` of sampler-specific arguments that fully characterize the
        desired sampler.  Refer to the documentation of each sampler for more
        information.

    Returns
    -------
    :
        The desired sampler function.
    """
    KEY = "expertMode"

    if sampler.lower() == "metropolis_hastings":
        return functools.partial(sample_with_metropolis_hastings, **options)
    elif sampler.upper() == "LMC":
        lmc_options = copy.deepcopy(options)

        if KEY in lmc_options:
            if not isinstance(lmc_options[KEY], bool):
                raise ValueError(f"{KEY} value must be a boolean")
            elif not lmc_options[KEY]:
                msg = "{} is included for unofficial research purposes only"
                raise ValueError(msg.format(sampler))

            del lmc_options[KEY]
        else:
            msg = "{} is included for unofficial research purposes only"
            raise ValueError(msg.format(sampler))

        # Emit warning to extend a helping hand to the experts.
        msg = f"Using unofficial research {sampler} sampler"
        warnings.warn(msg)
        return functools.partial(sample_with_LMC, **lmc_options)
    elif sampler.upper() == "PTLMC":
        return functools.partial(sample_with_PTLMC, **options)

    raise TypeError(f"Invalid sampler ({sampler})")
