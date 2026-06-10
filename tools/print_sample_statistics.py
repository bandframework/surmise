import numpy as np
import itertools as it

from estimate_sample_statistics import estimate_sample_statistics


def print_sample_statistics(distribution, samples):
    dim = distribution.dimension

    assert samples.ndim == 2
    assert samples.shape[1] == dim
    print(f'Number of samples: {samples.shape[0]}')

    if dim == 1:
        mean, var = distribution.moments
        assert var > 0.0
        mean_hat, var_hat = estimate_sample_statistics(samples)

        rel_err = np.fabs(1.0 - mean_hat / mean)
        print(
            f"Mean = {mean:<20} / "
            f"Sample mean = {mean_hat:<20} / "
            f"Rel err = {rel_err:<20}"
        )
        rel_err = np.fabs(1.0 - var_hat / var)
        print(
            f"Var  = {var:<20} / "
            f"Sample var  = {var_hat:<20} / "
            f"Rel err = {rel_err:<20}"
        )
        return

    mean, Cov = distribution.moments
    assert mean.ndim == 1
    assert len(mean) == dim
    assert Cov.ndim == 2
    assert Cov.shape == (dim, dim)
    assert np.array_equal(Cov, Cov.T)
    assert all(np.diag(Cov) > 0.0)
    mean_hat, Cov_hat = estimate_sample_statistics(samples)
    assert mean_hat.shape == mean.shape
    assert Cov_hat.shape == Cov.shape
    assert np.array_equal(Cov_hat, Cov_hat.T)

    for i, (mean_i, mean_i_hat) in enumerate(zip(mean, mean_hat)):
        rel_err = np.fabs(1.0 - mean_i_hat / mean_i)
        print(
            f"Mean X_{i+1}     = {mean_i:<20} / "
            f"Sample mean = {mean_i_hat:<20} / "
            f"Rel err = {rel_err:<20}"
        )

    for i, mean_i_hat in enumerate(mean_hat):
        var_i = Cov[i, i]
        var_i_hat = Cov_hat[i, i]
        rel_err = np.fabs(1.0 - var_i_hat / var_i)
        print(
            f"Var  X_{i+1}     = {var_i:<20} / "
            f"Sample var  = {var_i_hat:<20} / "
            f"Rel err = {rel_err:<20}"
        )

    for i, j in it.combinations(range(dim), 2):
        Cov_ij = Cov[i, j]
        Cov_ij_hat = Cov_hat[i, j]

        if np.fabs(Cov_ij) <= 1.0e-3:
            cov_err_text = "Abs err"
            cov_err = np.fabs(Cov_ij - Cov_ij_hat)
        else:
            cov_err_text = "Rel err"
            cov_err = np.fabs(1.0 - Cov_ij_hat / Cov_ij)
        print(
            f"Cov  X_{i+1}/X_{j+1} = {Cov_ij:<20} / "
            f"Sample Cov  = {Cov_ij_hat:<20} / "
            f"{cov_err_text} = {cov_err:<20}"
        )
