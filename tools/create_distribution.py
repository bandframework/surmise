import numbers

import numpy as np

from UniformDistribution import UniformDistribution
from JointUniformDistribution import JointUniformDistribution
from NormalDistribution import NormalDistribution


def create_distribution(configuration):
    name = configuration["Name"]
    if name.lower() == "uniform":
        ivals = np.atleast_2d(configuration["Intervals"])
        assert ivals.ndim == 2
        assert ivals.shape[1] == 2
        dimension = ivals.shape[0]
        if dimension == 1:
            a, b = ivals[0]
            distribution = UniformDistribution(a, b)
        else:
            # TODO: Allow for >2
            # distribution = JointUniformDistribution(*ivals)
            assert dimension == 2
            distribution = JointUniformDistribution(ivals[0], ivals[1])
    elif name.lower() == "normal":
        mu = configuration["mu"]
        dimension = 1 if isinstance(mu, numbers.Real) else len(mu)
        if dimension == 1:
            sigma = configuration["sigma"]
            distribution = NormalDistribution(mu, sigma**2)
        else:
            raise NotImplementedError("No ND normal yet")
    else:
        raise ValueError(f"Unknown distribution {name}")

    return distribution
