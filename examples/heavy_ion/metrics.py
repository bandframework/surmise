import numpy as np
import scipy.stats as sps
from scipy.linalg import sqrtm

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


def kl_divergence_gaussian(mu1, Cov1, mu2, Cov2):
    """
    Calculate the Kullback-Leibler divergence between two Gaussian distributions.
    
    Handles both univariate and multivariate cases by converting scalars to vectors and matrices.
    
    Parameters:
        mu1 (float or np.ndarray): Mean of the first Gaussian distribution.
        Cov1 (float or np.ndarray): Covariance matrix (univariate as scalar or multivariate as matrix) of the first Gaussian distribution.
        mu2 (float or np.ndarray): Mean of the second Gaussian distribution.
        Cov2 (float or np.ndarray): Covariance matrix (univariate as scalar or multivariate as matrix) of the second Gaussian distribution.
    
    Returns:
        float: The KL divergence D_KL(N_1 || N_2).
    """
    # Convert scalars to 1D vectors for means
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    # Convert scalars to 2D matrices for covariances
    Cov1 = np.atleast_2d(Cov1)
    Cov2 = np.atleast_2d(Cov2)
    
    # Ensure dimensions match
    if Cov1.shape != Cov2.shape or mu1.shape != mu2.shape:
        raise ValueError("Mean vectors and covariance matrices must have the same dimensions.")
    
    # Ensure covariance matrices are positive definite
    if not np.all(np.linalg.eigvals(Cov1) > 0) or not np.all(np.linalg.eigvals(Cov2) > 0):
        raise ValueError("Covariance matrices must be positive definite.")
    
    # Dimensionality of the distributions
    k = mu1.shape[0]
    
    # Compute the KL divergence
    Cov2_inv = np.linalg.inv(Cov2)
    term1 = np.trace(Cov2_inv @ Cov1)
    term2 = (mu2 - mu1).T @ Cov2_inv @ (mu2 - mu1)
    term3 = np.log(np.linalg.det(Cov2) / np.linalg.det(Cov1))
    
    kl_div = 0.5 * (term1 + term2 - k + term3)
    return kl_div


def hellinger_distance_gaussian(mu1, Cov1, mu2, Cov2):
    """
    Calculate the Hellinger distance between two Gaussian distributions.
    
    Handles both univariate and multivariate cases by converting scalars to vectors and matrices.
    
    Parameters:
        mu1 (float or np.ndarray): Mean of the first Gaussian distribution.
        Cov1 (float or np.ndarray): Covariance matrix (univariate as scalar or multivariate as matrix) of the first Gaussian distribution.
        mu2 (float or np.ndarray): Mean of the second Gaussian distribution.
        Cov2 (float or np.ndarray): Covariance matrix (univariate as scalar or multivariate as matrix) of the second Gaussian distribution.
    
    Returns:
        float: The Hellinger distance H(N_1, N_2).
    """
    # Convert scalars to 1D vectors for means
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    # Convert scalars to 2D matrices for covariances
    Cov1 = np.atleast_2d(Cov1)
    Cov2 = np.atleast_2d(Cov2)
    
    # Ensure dimensions match
    if Cov1.shape != Cov2.shape or mu1.shape != mu2.shape:
        raise ValueError("Mean vectors and covariance matrices must have the same dimensions.")
    
    # Compute the mean and covariance terms
    diff_mu = mu1 - mu2
    Cov_avg = 0.5 * (Cov1 + Cov2)
    
    # Ensure Cov_avg is positive definite
    if not np.all(np.linalg.eigvals(Cov_avg) > 0):
        raise ValueError("Covariance matrices must be positive definite.")
        
    # Compute the determinant term
    det_term = np.sqrt( np.sqrt(np.linalg.det(Cov1) * np.linalg.det(Cov2)) / np.linalg.det(Cov_avg) )
    
    # Compute the exponential term
    exp_term = np.exp(-0.125 * diff_mu.T @ np.linalg.inv(Cov_avg) @ diff_mu)
    
    # Hellinger distance squared
    H2 = 1 - det_term * exp_term
    
    # Hellinger distance
    H = np.sqrt(H2)
    
    return H


def wasserstein_distance_gaussian(mu1, Cov1, mu2, Cov2):
    """
    Calculate the 2-Wasserstein distance between two Gaussian distributions.
    
    Handles both univariate and multivariate cases by converting scalars to vectors and matrices.
    
    Parameters:
        mu1 (float or np.ndarray): Mean of the first Gaussian distribution.
        Cov1 (float or np.ndarray): Covariance matrix (or variance in univariate case) of the first Gaussian distribution.
        mu2 (float or np.ndarray): Mean of the second Gaussian distribution.
        Cov2 (float or np.ndarray): Covariance matrix (or variance in univariate case) of the second Gaussian distribution.
    
    Returns:
        float: The 2-Wasserstein distance W_2(N_1, N_2).
    """
    # Convert scalars to 1D vectors for means
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    # Convert scalars to 2D matrices for covariances (variances in univariate case)
    Cov1 = np.atleast_2d(Cov1)
    Cov2 = np.atleast_2d(Cov2)
    
    # Compute the squared Euclidean distance between the means
    mean_diff_sq = np.sum((mu1 - mu2) ** 2)
    
    # Compute the square root of Cov1
    sqrt_Cov1 = sqrtm(Cov1)
    
    # Compute the square root of the product Cov1^{1/2} @ Cov2 @ Cov1^{1/2}
    sqrt_product = sqrtm(sqrt_Cov1 @ Cov2 @ sqrt_Cov1)
    
    # Ensure the result is real (sometimes sqrtm can introduce a small imaginary component due to numerical precision)
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real
    
    # Compute the trace term
    trace_term = np.trace(Cov1 + Cov2 - 2 * sqrt_product)
    
    # Wasserstein distance
    W2 = np.sqrt(mean_diff_sq + trace_term)
    
    return W2


# Simplified for univariate gaussians. can be used for testing -->

# def kl_divergence_1Dgaussian(mu1, sigma1, mu2, sigma2):
#     """
#     Returns KL Divergence for two univarite Gaussian distributions
#     mu1, mu2 (float): Mean of the Gaussian distributions.
#     sigma1, sigma2 (float): standard deviations of the Gaussian distribution.
#     """
#     return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2.0 * sigma2**2) - 0.5


# def hellinger_distance_1Dgaussian(mu1, sigma1, mu2, sigma2):
#     """
#     Returns Hellinger Distance (bounded metric between 0 and 1) for two Gaussian distributions
#     mu1, mu2 (float): Mean of the Gaussian distributions.
#     sigma1, sigma2 (float): standard deviations of the Gaussian distribution.
#     """
#     term1 = np.sqrt(2.0 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
#     term2 = np.exp(-0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2))
#     return np.sqrt(1.0 - term1 * term2)


# def wasserstein_distance_1Dgaussian(mu1, sigma1, mu2, sigma2):
#     """
#     Returns Wasserstein Distance (or earth mover's distance) for two Gaussian distributions
#     mu1, mu2 (float): Mean of the Gaussian distributions.
#     sigma1, sigma2 (float): standard deviations of the Gaussian distribution.
#     """
#     return np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)





