import numbers

import numpy as np
import scipy.stats as sps

from AbstractDistribution import AbstractDistribution


class UniformDistribution(AbstractDistribution):
    def __init__(self, a, b, i=None):
        """
        Uniform distribution on closed interval [a, b].
        """
        super().__init__()

        if (not isinstance(a, numbers.Real)) or (not np.isfinite(a)):
            raise ValueError("a must be finite real")
        if (not isinstance(b, numbers.Real)) or (not np.isfinite(b)):
            raise ValueError("b must be finite real")

        length = b - a
        if length <= 0.0:
            raise ValueError("b <= a is not a valid interval")

        self.__U = sps.uniform(loc=a, scale=length)

        if i is None:
            print(f"[a, b] = [{a}, {b}]")
        else:
            print(f"[a_{i}, b_{i}] = [{a}, {b}]")

    @property
    def dimension(self):
        return 1

    @property
    def moments(self):
        return self.__U.stats("mv")

    def inv_cdf(self, p):
        return self.__U.ppf(self._as2darray_checked(p))

    def pdf(self, theta):
        return self.__U.pdf(self._as2darray_checked(theta))

    def logpdf(self, theta, return_grad):
        if return_grad:
            raise NotImplementedError("gradient not implemented yet")
        return self.__U.logpdf(self._as2darray_checked(theta))

    def sample(self, n, rng):
        samples = np.atleast_2d(self.__U.rvs(size=n, random_state=rng))
        assert samples.shape == (1, n)
        return samples.T
