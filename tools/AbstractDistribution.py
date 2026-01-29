import abc

import numpy as np


class AbstractDistribution(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def _as2darray_checked(self, theta):
        """
        :return: real, finite 2d numpy array where rows correspond to theta
            points; columns, to theta coordinates.
        """
        checked = np.squeeze(np.asarray_chkfinite(theta))
        if not all(np.isreal(checked.flatten())):
            raise ValueError("theta values are not real")

        if self.dimension == 1:
            checked = np.atleast_2d(checked).T
            assert checked.shape[1] == 1
            return checked
        elif checked.ndim != 2:
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
        :return: (mean, variance)
        """
        ...

    @abc.abstractmethod
    def pdf(self, theta):
        ...

    @abc.abstractmethod
    def logpdf(self, theta, return_grad):
        ...

    @abc.abstractmethod
    def sample(self, n, rng):
        """
        :param rng: must be scipy.stats-compatible RNG
        :return: :math:`n` independent samples from distribution as 2D numpy
            array with each row corresponding to a sample.
        """
        ...
