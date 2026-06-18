import numbers

import numpy as np

from UniformDistribution import UniformDistribution
from JointUniformDistribution import JointUniformDistribution
from NormalDistribution import NormalDistribution
from MultinormalDistribution import MultinormalDistribution


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
            distribution = JointUniformDistribution(*ivals)
    elif name.lower() == "normal":
        mu = configuration["mu"]
        dimension = 1 if isinstance(mu, numbers.Real) else len(mu)
        if dimension == 1:
            sigma = configuration["sigma"]
            distribution = NormalDistribution(mu, sigma**2)
        else:
            mu = np.array(mu)
            Cov = np.array(configuration["sigma"])
            distribution = MultinormalDistribution(mu, Cov)
    else:
        raise ValueError(f"Unknown distribution {name}")

    return distribution
