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
        mean, var = self.moments
        assert mean == mu
        assert var == sigma_sqr

        print(f"Mean\t\t\t{mean}")
        print(f"Variance\t\t{var}")
        print(f"Standard deviation\t{self.__N.std()}")

    @property
    def dimension(self):
        return 1

    @property
    def moments(self):
        return self.__N.stats("mv")

    def inv_cdf(self, p):
        values = self.__N.ppf(self._as1darray_checked(p))
        assert values.ndim == 1
        return values

    def pdf(self, theta):
        values = self.__N.pdf(self._as1darray_checked(theta))
        assert values.ndim == 1
        return values

    def logpdf(self, theta, return_grad=False):
        if return_grad:
            raise NotImplementedError("gradient not implemented yet")
        values = self.__N.logpdf(self._as1darray_checked(theta))
        assert values.ndim == 1
        return values

    def marginal_pdf(self, _):
        raise NotImplementedError("No need for marginals with 1D distributions")

    def sample(self, n, rng):
        samples = np.atleast_2d(self.__N.rvs(size=n, random_state=rng))
        assert samples.shape == (1, n)
        return samples.T
