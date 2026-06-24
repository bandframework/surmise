import numpy as np
import itertools as it

from approximate_integral import approximate_integral


def estimate_sample_statistics(samples):
    assert samples.ndim == 2
    n_samples, dim = samples.shape
    assert n_samples > 1

    if dim == 1:
        samples_1d = np.squeeze(samples)

        mean_hat = approximate_integral(lambda x: x, samples_1d)
        sqr_hat = approximate_integral(lambda x: x**2, samples_1d)
        var_hat = sqr_hat - mean_hat**2

        return mean_hat, var_hat

    mean_hat = np.full(dim, np.nan, float)
    Cov_hat = np.full([dim, dim], np.nan, float)
    for i in range(dim):
        mean_i_hat = approximate_integral(lambda x: x, samples[:, i])
        sqr_i_hat = approximate_integral(lambda x: x**2, samples[:, i])
        mean_hat[i] = mean_i_hat
        Cov_hat[i, i] = sqr_i_hat - mean_i_hat**2

    for i, j in it.combinations(range(dim), 2):
        prod_ij_hat = approximate_integral(lambda x: x,
                                           samples[:, i] * samples[:, j])
        Cov_hat[i, j] = prod_ij_hat - mean_hat[i] * mean_hat[j]
        Cov_hat[j, i] = Cov_hat[i, j]

    return mean_hat, Cov_hat
