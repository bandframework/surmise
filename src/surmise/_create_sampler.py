import functools

from ._samplermethods import (
    sample_with_metropolis_hastings, sample_with_LMC,
    sample_with_bilby
)


def create_sampler(description, options=None):
    """
    It is intended that this function only be called by Calibrators.  Therefore,
    it should not be in the public interface.
    """
    if isinstance(description, str):
        if description.lower() == "metropolis_hastings":
            return functools.partial(
                sample_with_metropolis_hastings,
                options=options
            )
        elif description.upper() == "LMC":
            return functools.partial(
                sample_with_LMC,
                options=options
            )
    elif isinstance(description, dict):
        assert len(description) == 1
        source = list(description)[0]
        if source.lower() == "bilby":
            return functools.partial(
                sample_with_bilby,
                sampler=description[source],
                options=options
            )
        elif source.lower() == "user":
            raise NotImplementedError("Cannot create user-provided samplers yet")
    else:
        raise TypeError(f"Invalid description type ({description})")

    raise ValueError(f"Invalid sampler description ({description})")
