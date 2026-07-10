import numpy as np


def create_scipy_stats_rng(rng_cfg):
    rand_method = rng_cfg["method"]
    rand_seed = rng_cfg["random_seed"]
    assert rand_method.lower() == "default"

    print(f"RNG method\t\t{rand_method}")
    print(f"Random seed\t\t{rand_seed}")

    return np.random.default_rng(rand_seed)
