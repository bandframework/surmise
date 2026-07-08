.. _rng_user_guide:

Random Number Generation
========================
Following typical practices, we refer to pseudorandom number generation and
generators more generically as random number generation and random number
generators (RNGs).

|surmise| code uses exclusively the ``scipy.stats`` code to sample all random
numbers and for performing typical statistical computations.  At any point in
time the code uses only a single user-provided ``scipy.stats``-compatible RNG to
sample random numbers.  Therefore, before calling |surmise| code, users must
provide |surmise| with an RNG that is valid for their version of ``scipy`` as
well as correctly created and managed for their application.  Note that where
possible all |surmise| code should reproduce the same results when the same task
is run with an identical RNG setup.

The following demonstrates this and shows that users are free to change the
single RNG being used by |surmise|.

.. code:: python

    import secrets
    import surmise
    import numpy as np

    RAND_SEED = secrets.randbits(128)

    surmise.set_RNG(np.random.default_rng(RAND_SEED))
    samples_1 = surmise.calibration().calibration_samples
    samples_2 = surmise.calibration().calibration_samples
    assert not all(samples_1 == samples_2)

    surmise.set_RNG(np.random.default_rng(RAND_SEED))
    samples_3 = surmise.calibration().calibration_samples
    assert all(samples_1 == samples_3)

..
    External code offered officially through |surmise|, such as |bilby|, have
    their own RNG usage scheme that is independent from the |surmise| scheme.
    In particular, the RNG provided to |surmise| is never used explicitly by
    external code.  Instead, users are responsible for understanding the
    external code's RNG scheme within the context of the application's needs and
    providing additional RNG configuration information to |surmise| code that
    uses the external code.

..
    Please refer to the RNG examples in the Jupyter book for more examples of
    using RNGs with |surmise| including the RNG configuration of external code.
