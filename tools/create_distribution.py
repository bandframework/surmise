import numpy as np

from UniformDistribution import UniformDistribution


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
            raise NotImplementedError("No ND uniform yet")
    else:
        raise ValueError(f"Unknown distribution {name}")

    return distribution
