import numpy as np


class _RngSingleton(type):
    __objs = {}

    def __call__(cls):
        if cls not in cls.__objs:
            cls.__objs[cls] = super(_RngSingleton, cls).__call__()
        return cls.__objs[cls]


class RandomNumberGenerator(metaclass=_RngSingleton):
    def __init__(self):
        """
        This class is implemented using the Singleton design pattern and
        therefore enforces the design decision that at most only one
        ``RandomNumberGenerator`` object, and therefore one ``scipy.stats`` RNG,
        can exist at a time.  In addition, once that instance has been created,
        it will persist through program execution.  However, users are allowed
        to change the single ``scipy.stats`` RNG managed by that instance as
        many times as desired and when desired.

        Any |surmise| code can access the single ``RandomNumberGenerator``
        object permitted by this Singleton class using

        .. code-block:: python

            from ._RandomNumberGenerator import RandomNumberGenerator

            global_rng = RandomNumberGenerator().scipy_stats_RNG
        """
        # We do not set a default RNG upon instantiation so that we do not
        # implicitly assume responsibility for constructing a correct, default
        # scipy.stats RNG.  While this puts the responsiblity of determining how
        # to do this on the user, it ensures that this class isn't accidentally
        # constructing an RNG that is out of date for the user's scipy
        # installation.  Rather the user can always provide an RNG that is valid
        # for their version of scipy.stats and the rest of the surmise code will
        # use it correctly so long as the rest of the scipy.stats interface has
        # not changed significantly.
        self.__rng = None

    @property
    def scipy_stats_RNG(self):
        """
        |surmise| internal code should never store the RNG obtained with this for
        later use (e.g., in a class's constructor).  Rather upon each invocation,
        the internal code shall use this member function to access the current RNG
        set into |surmise|.

        An exception is raised if the RNG has not yet been set by users.

        Returns
        -------
        :
            Current global RNG to be used by all |surmise| code with
            ``scipy.stats`` for all random number generation
        """
        if self.__rng is None:
            raise RuntimeError("Please use set_RNG before using surmise")
        return self.__rng

    @scipy_stats_RNG.setter
    def scipy_stats_RNG(self, rng):
        """
        This should **only** be called indirectly by users |via| ``set_RNG`` and
        **never** by |surmise| internal code.

        Parameters
        ----------
        rng :
            ``scipy.stats``-compatible RNG that all |surmise| code should use
            for all random number generation
        """
        # Check general design assumptions
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Given RNG cannot be used with scipy.stats")
        elif (not hasattr(rng, "choice")) or \
                (not callable(getattr(rng, "choice"))):
            raise RuntimeError("Given RNG does not provide the choice function")

        self.__rng = rng

    def _clear_RNG(self):
        """Testing support only. Returns singleton to its unset state. This is not intended for user manipulation
        of the RNG."""
        self.__rng = None