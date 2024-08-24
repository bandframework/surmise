import numpy as np
import scipy.stats as sps


def rmse(y, ypredmean):
    """
    Returns root mean squared error.
    """
    return np.sqrt(np.mean((y - ypredmean) ** 2))


def normalized_rmse(y, ypredmean):
    """
    Returns normalized root mean squared error, error normalized by range for each
    output dimension.
    """
    rng = (np.max(y, axis=1) - np.min(y, axis=1)).reshape(y.shape[0], 1)
    return np.sqrt(np.mean(((y - ypredmean) / rng)**2))


def dss(y, ypredmean, ypredcov, use_diag):
    """
    Returns Dawid-Sebastiani score from Gneiting et al. (2007) Eq. 25.
    """
    def __dss_single(f, mu, Sigma):
        r = f - mu
        W, U = np.linalg.eigh(Sigma)
        r_Sinvh = r @ U * 1 / np.sqrt(W)

        _, logabsdet = np.linalg.slogdet(Sigma)

        score_single = logabsdet + (r_Sinvh ** 2).sum()
        return score_single

    def __dss_single_diag(f, mu, diagSigma):
        r = f - mu
        score_single = np.log(diagSigma).sum() + (r * r / diagSigma).sum()
        return score_single

    p, n = y.shape
    score = 0
    if use_diag:
        for i in range(n):
            score += __dss_single_diag(y[:, i], ypredmean[:, i], ypredcov[:, i])
    else:
        for i in range(n):
            score += __dss_single(y[:, i], ypredmean[:, i], ypredcov[:, :, i])
    score /= n

    return score


def intervalstats(y, ypredmean, ypredvar):
    """
    Returns empirical coverage and length of interval given true/observed $y$,
    predictive means and variances.
    """
    ylower = ypredmean + np.sqrt(ypredvar) * sps.norm.ppf(0.025)
    yupper = ypredmean + np.sqrt(ypredvar) * sps.norm.ppf(0.975)

    coverage = np.mean(np.logical_and(y <= yupper, y >= ylower))
    length = np.mean(yupper - ylower)
    return coverage, length


def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Returns KL Divergence for two Gaussian distributions
    """
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2.0 * sigma2**2) - 0.5

def hellinger_distance_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Returns Hellinger Distance (bounded metric between 0 and 1) for two Gaussian distributions
    """
    term1 = np.sqrt(2.0 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
    term2 = np.exp(-0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2))
    return np.sqrt(1.0 - term1 * term2)

def wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Returns Wasserstein Distance (or earth mover's distance) for two Gaussian distributions
    """
    return np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)





