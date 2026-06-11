import numpy as np

from AbstractDistribution import AbstractDistribution


class IndependentJointDistribution(AbstractDistribution):
    def __init__(self, univariate_distributions):
        """
        This is poorly named, but anything else is ludicrously long.
        """
        self.__univariates = univariate_distributions
        for distribution_i in self.__univariates:
            if not isinstance(distribution_i, AbstractDistribution):
                raise TypeError(
                    "Univariate distribution is not an AbstractDistribution"
                )
            elif distribution_i.dimension != 1:
                raise ValueError(
                    "Univariate distribution must actually be univariate"
                )

        if self.dimension < 2:
            raise ValueError(
                "Need at least two independent univariate distributions"
            )

    @property
    def dimension(self):
        return len(self.__univariates)

    @property
    def moments(self):
        mu = np.full(self.dimension, np.nan, float)
        sigma_sqr = mu.copy()
        for i, distribution_i in enumerate(self.__univariates):
            mu[i], sigma_sqr[i] = distribution_i.moments
        assert all(np.isreal(mu)) and all(np.isfinite(mu))
        assert all(np.isreal(sigma_sqr)) and all(np.isfinite(sigma_sqr))
        assert all(sigma_sqr > 0.0)
        return mu, np.diag(sigma_sqr)

    def inv_cdf(self, p):
        quantiles = np.full((self.dimension, len(p)), np.nan, float)
        for i, distribution_i in enumerate(self.__univariates):
            quantiles[i, :] = distribution_i.inv_cdf(p).reshape(-1)
        assert all(np.isreal(quantiles.ravel()))
        assert all(np.isfinite(quantiles.ravel()))
        return quantiles

    def pdf(self, theta):
        theta_2d = self._as2darray_checked(theta)
        values = np.ones(theta_2d.shape[0])
        for i, distribution_i in enumerate(self.__univariates):
            values *= distribution_i.pdf(theta_2d[:, i])
        assert all(np.isreal(values)) and all(np.isfinite(values))
        assert all(values >= 0.0)
        return values

    def logpdf(self, theta, return_grad):
        if return_grad:
            raise NotImplementedError("gradient not implemented")

        theta_2d = self._as2darray_checked(theta)
        values = np.zeros(theta_2d.shape[0])
        for i, distribution_i in enumerate(self.__univariates):
            values += distribution_i.logpdf(theta_2d[:, i], return_grad)
        assert all(np.isreal(values)) and (not any(np.isnan(values)))
        assert not any(values == np.inf)
        return values

    def sample(self, n, rng):
        samples = np.full([n, self.dimension], np.nan, float)
        for i, distribution_i in enumerate(self.__univariates):
            samples[:, i] = np.squeeze(distribution_i.sample(n, rng))
        assert all(np.isreal(samples.flatten()))
        assert all(np.isfinite(samples.flatten()))
        return samples
