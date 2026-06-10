import numpy as np

from approximate_integral import approximate_integral


def print_sample_statistics(distribution, samples):
    dim = distribution.dimension

    assert samples.ndim == 2
    n_samples = samples.shape[0]
    assert n_samples > 1
    assert samples.shape[1] == dim
    print(f'Number of samples: {n_samples}')

    if dim == 1:
        samples_1d = np.squeeze(samples)

        mean, var = distribution.moments
        assert var > 0.0

        mean_hat = approximate_integral(lambda x: x, samples_1d)
        sqr_hat = approximate_integral(lambda x: x**2, samples_1d)
        var_hat = sqr_hat - mean_hat**2

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

    for i, mean_i in enumerate(mean):
        mean_i_hat = approximate_integral(lambda x: x, samples[:, i])

        rel_err = np.fabs(1.0 - mean_i_hat / mean_i)
        print(
            f"Mean X_{i+1} = {mean_i:<20} / "
            f"Sample mean = {mean_i_hat:<20} / "
            f"Rel err = {rel_err:<20}"
        )

    # TODO: Print out variances and covariances
