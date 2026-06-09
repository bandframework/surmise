import numbers

import numpy as np
import scipy.stats as sps

from AbstractDistribution import AbstractDistribution


class NormalDistribution(AbstractDistribution):
    def __init__(self, mu, sigma_sqr):
        """
        Normal distribution with mean mu and variance sigma_sqr
        """
        if (not isinstance(mu, numbers.Real)) or (not np.isfinite(mu)):
            raise ValueError("mu must be finite real")
        if (not isinstance(sigma_sqr, numbers.Real)) \
                or (not np.isfinite(sigma_sqr)):
            raise ValueError("sigma_sqr must be finite real")
        elif sigma_sqr <= 0.0:
            raise ValueError("sigma_sqr must be positive")

        self.__N = sps.norm(loc=mu, scale=np.sqrt(sigma_sqr))

        print(f"Mean\t\t\t{mu}")
        print(f"Variance\t\t{sigma_sqr}")

    @property
    def dimension(self):
        return 1

    @property
    def moments(self):
        return self.__N.stats("mv")

    def inv_cdf(self, p):
        return self.__N.ppf(self._as2darray_checked(p))

    def pdf(self, theta):
        return self.__N.pdf(self._as2darray_checked(theta))

    def logpdf(self, theta, return_grad):
        if return_grad:
            raise NotImplementedError("gradient not implemented yet")
        return self.__N.logpdf(self._as2darray_checked(theta))

    def sample(self, n, rng):
        samples = np.atleast_2d(self.__N.rvs(size=n, random_state=rng))
        assert samples.shape == (1, n)
        return samples.T
