"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    
    # Extract the parameters from the mixture
    mu = mixture.mu  # (K, d)
    var = mixture.var  # (K,)
    p = mixture.p  # (K,)
    
    # Initialize array to hold log-probabilities for each data point and component
    log_post = np.zeros((n, K))
    
    # Compute log-probabilities for each data point and component
    for j in range(K):
        # Compute the log of the multivariate normal density
        diff = X - mu[j]  # (n, d)
        norm_sq = np.sum(diff**2, axis=1)  # (n,)
        log_normal_density = -norm_sq / (2 * var[j]) - d/2 * np.log(2 * np.pi * var[j])  # (n,)
        
        # Add the log of the mixing proportion
        log_post[:, j] = np.log(p[j]) + log_normal_density  # (n,)
    
    # Use the log-sum-exp trick to compute the log of the marginal likelihood
    max_log = np.max(log_post, axis=1, keepdims=True)  # (n, 1)
    log_marginal = max_log + np.log(np.sum(np.exp(log_post - max_log), axis=1, keepdims=True))  # (n, 1)
    
    # Compute the posterior probabilities
    post = np.exp(log_post - log_marginal)  # (n, K)
    
    # Compute the log-likelihood
    ll = np.sum(log_marginal)  # scalar
    
    return post, ll




def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    # Compute the effective number of points assigned to each cluster
    Nk = np.sum(post, axis=0)  # shape (K,)

    # Update means: μₖ = (1/Nₖ) * Σᵢ γ(z^(i)ₖ) * x^(i)
    mu = (post.T @ X) / Nk[:, np.newaxis]  # shape (K, d)

    # Update variances: σₖ² = (1/(d*Nₖ)) * Σᵢ γ(z^(i)ₖ) * ||x^(i) - μₖ||²
    var = np.zeros(K)
    for k in range(K):
        diff = X - mu[k]  # shape (n, d)
        sq_dist = np.sum(diff**2, axis=1)  # shape (n,)
        var[k] = np.sum(post[:, k] * sq_dist) / (Nk[k] * d)

    # Update mixing proportions: πₖ = Nₖ / n
    p = Nk / n  # shape (K,)

    return GaussianMixture(mu, var, p)



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
