import numpy as np


def create_scipy_stats_rng(rng_cfg):
    rand_method = rng_cfg["method"]
    rand_seed = rng_cfg["random_seed"]

    print(f"RNG method\t\t{rand_method}")
    print(f"Random seed\t\t{rand_seed}")

    if rand_method.lower() == "default":
        return np.random.default_rng(rand_seed)
    elif rand_method.upper() == "PCG64DXSM":
        return np.random.Generator(np.random.PCG64DXSM(rand_seed))

    raise ValueError(f"Unsupported bit generator ({rand_method})")
