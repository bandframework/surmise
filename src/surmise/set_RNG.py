from ._RandomNumberGenerator import RandomNumberGenerator


def set_RNG(scipy_stats_rng):
    """
    Prior to using any |surmise| functionality, users should call this function
    to provide |surmise| with a single pseudo-random number generator for use
    with their version of ``scipy``.

    Parameters
    ----------
    scipy_stats_rng :
        RNG that all |surmise| code uses to sample random numbers with
        ``scipy.stats``
    """
    RandomNumberGenerator().scipy_stats_RNG = scipy_stats_rng
