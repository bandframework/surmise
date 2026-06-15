import numbers

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
        # Marginal quantiles
        quantiles = np.full((self.dimension, len(p)), np.nan, float)
        for i in range(self.dimension):
            quantiles[i, :] = self._as1darray_checked(
                sps.norm.ppf(p, loc=self.__N.mean[i],
                             scale=np.sqrt(self.__N.cov[i, i])))
        return quantiles

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

    def marginal_pdf(self, theta, index):
        if isinstance(index, numbers.Integral):
            if (index < 0) or (index >= self.dimension):
                raise ValueError(f"Invalid random variable index ({index})")
            values = sps.norm.pdf(self._as1darray_checked(theta),
                                  loc=self.__N.mean[index],
                                  scale=np.sqrt(self.__N.cov[index, index]))
        elif isinstance(index, (list, tuple)):
            # TODO: Asserts should ideally be error checks that raise exceptions
            indices = self._as1darray_checked(index)
            assert len(indices) == 2
            i_0, i_1 = indices
            assert (0 <= i_0) and (i_0 < self.dimension)
            assert (0 <= i_1) and (i_1 < self.dimension)
            assert i_0 != i_1
            mean = np.array([self.__N.mean[i_0], self.__N.mean[i_1]])
            Cov = np.array([
                [self.__N.cov[i_0, i_0], self.__N.cov[i_0, i_1]],
                [self.__N.cov[i_1, i_0], self.__N.cov[i_1, i_1]]
            ])
            checked = np.atleast_2d(np.squeeze(np.asarray_chkfinite(theta)))
            assert checked.ndim == 2
            assert checked.shape[1] == len(indices)
            assert all(np.isreal(checked.ravel()))
            values = sps.multivariate_normal.pdf(
                checked,
                mean=mean, cov=Cov,
                allow_singular=False
            )
        else:
            raise NotImplementedError(f"Invalid marginal indices ({index})")
        assert all(np.isreal(values)) and all(np.isfinite(values))
        assert all(values >= 0.0)
        return values

    def sample(self, n, rng):
        samples = np.atleast_2d(self.__N.rvs(size=n, random_state=rng))
        assert samples.shape == (n, self.dimension)
        return samples
