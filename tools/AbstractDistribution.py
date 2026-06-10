import abc

import numpy as np


class AbstractDistribution(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def _as1darray_checked(self, x):
        """
        :return: real, finite 1d numpy array
        """
        checked = np.atleast_1d(np.squeeze(np.asarray_chkfinite(x)))
        if not all(np.isreal(checked.flatten())):
            raise ValueError("x values are not real")
        if checked.ndim != 1:
            raise ValueError("x must be 1D array")

        return checked

    def _as2darray_checked(self, theta):
        """
        :return: real, finite 2d numpy array where rows correspond to theta
            points; columns, to theta coordinates.
        """
        checked = np.atleast_2d(np.squeeze(np.asarray_chkfinite(theta)))
        if not all(np.isreal(checked.flatten())):
            raise ValueError("theta values are not real")

        if self.dimension == 1:
            checked = checked.T
            assert checked.shape[1] == 1
            return checked

        if checked.ndim != 2:
            raise ValueError("theta must be 2D array")
        elif checked.shape[1] != self.dimension:
            raise ValueError("theta array has invalid shape")

        return checked

    @abc.abstractproperty
    def dimension(self):
        ...

    @abc.abstractproperty
    def moments(self):
        """
        :return: ``(mean, Sigma)`` where
            * ``mean`` is a scalar for a 1D problem and a 1D numpy array
              containing the mean of each parameter otherwise and
            * ``Sigma`` is the variance as a scalar for a 1D problem and a
              square, 2D numpy array containing the variances and covariances
              otherwise
            For >1D, the ordering of the array and matrix match the ordering
            provided by calling code during construction of the distribution
        """
        ...

    def inv_cdf(self, p):
        """
        :param p: 1D numpy array of probability values, must be between 0 and 1.
        :return: 1D numpy array of quantile values of the distribution.
        """
        raise NotImplementedError("Inverse cdf inv_cdf is not implemented.")

    @abc.abstractmethod
    def pdf(self, theta):
        """
        :param theta: 2D numpy array of points at which to evaluate the PDF.
            Each row should correspond to a single point; each column, to a
            different parameter.  The column order should match the ordering
            provided by calling code during construction of the distribution.
        :return: 1D numpy array of values
        """
        ...

    @abc.abstractmethod
    def logpdf(self, theta, return_grad):
        """
        :param theta: 2D numpy array of points at which to evaluate the log of
            the PDF.  Each row should correspond to a single point; each column,
            to a different parameter.  The column order should match the
            ordering provided by calling code during construction of the
            distribution.
        :return: 1D numpy array of values
        """
        ...

    @abc.abstractmethod
    def sample(self, n, rng):
        """
        :param rng: must be scipy.stats-compatible RNG
        :return: :math:`n` independent samples from distribution as 2D numpy
            array with each row corresponding to a sample.  For >1D, the column
            ordering matches the ordering provided by calling code during
            construction of the distribution
        """
        ...
