import numpy as np
import scipy.stats as sps

from AbstractDistribution import AbstractDistribution


class MultinormalDistribution(AbstractDistribution):
    # TODO: Combine this with NormalDistribution?
    def __init__(self, mean, Cov):
        assert mean.ndim == 1
        dim = len(mean)
        assert dim > 1
        assert Cov.ndim == 2
        assert Cov.shape == (dim, dim)
        assert np.array_equal(Cov, Cov.T)

        self.__N = sps.multivariate_normal(mean=mean, cov=Cov,
                                           allow_singular=False)
        mu, Sigma = self.moments
        assert np.array_equal(mean, mu)
        assert np.array_equal(Sigma, Cov)

        print(f"Mean\t\t\t{mu}")
        print(f"Covariance Matrix\n{Sigma}")

    @property
    def dimension(self):
        return len(self.__N.mean)

    @property
    def moments(self):
        return self.__N.mean, self.__N.cov

    def inv_cdf(self, p):
        # TODO: Moses to code this up
        raise NotImplementedError("This still needs some work")

    def pdf(self, theta):
        values = self.__N.pdf(self._as2darray_checked(theta))
        assert values.ndim == 1
        assert all(np.isreal(values)) and all(np.isfinite(values))
        assert all(values >= 0.0)
        return values

    def logpdf(self, theta, return_grad):
        if return_grad:
            raise NotImplementedError("gradient not implemented")

        values = np.atleast_1d(self.__N.logpdf(self._as2darray_checked(theta)))
        assert values.ndim == 1
        assert all(np.isreal(values)) and (not any(np.isnan(values)))
        assert not any(values == np.inf)
        return values

    def marginal_pdf(self, _):
        # TODO: Moses to code this up
        raise NotImplementedError("This still needs some work")

    def sample(self, n, rng):
        samples = np.atleast_2d(self.__N.rvs(size=n, random_state=rng))
        assert samples.shape == (n, self.dimension)
        return samples
