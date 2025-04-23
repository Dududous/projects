"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    
    # Create mask for observed entries: 1 if observed, 0 if missing
    observed_mask = (X != 0).astype(float)
    
    # Initialize log posterior matrix
    log_post = np.zeros((n, K))
    
    for j in range(K):
        # Extract parameters for cluster j
        mu_j = mixture.mu[j]
        var_j = mixture.var[j]
        p_j = mixture.p[j]
        
        # Calculate squared difference for observed dimensions
        diff = X - mu_j
        diff = diff * observed_mask  # Zero out differences for missing dimensions
        sq_diff = np.sum(diff**2, axis=1)
        
        # Count observed dimensions per data point
        n_obs_dims = np.sum(observed_mask, axis=1)
        
        # Log Gaussian density for observed dimensions
        log_norm = -0.5 * n_obs_dims * np.log(2 * np.pi * var_j) - 0.5 * sq_diff / var_j
        
        # Log posterior = log prior + log likelihood
        log_post[:, j] = np.log(p_j) + log_norm
    
    # Use log-sum-exp trick for numerical stability
    log_post_max = np.max(log_post, axis=1, keepdims=True)
    log_sum = log_post_max + np.log(np.sum(np.exp(log_post - log_post_max), axis=1, keepdims=True))
    
    # Log-likelihood of the data
    ll = np.sum(log_sum)
    
    # Compute posterior probabilities (soft assignments)
    post = np.exp(log_post - log_sum)
    
    return post, ll



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    
    # Create mask for observed entries (1 if observed, 0 if missing)
    observed = (X != 0).astype(float)
    
    # Update mixing coefficients (π_k)
    p = np.sum(post, axis=0) / n
    
    # Initialize new means with current values
    mu = mixture.mu.copy()
    var = np.zeros(K)
    
    # Update means (μ)
    for k in range(K):
        for l in range(d):
            # Calculate weights for dimension l in component k
            weights = post[:, k] * observed[:, l]
            weight_sum = np.sum(weights)
            
            # Only update mean if we have sufficient support (sum of weights ≥ 1)
            if weight_sum >= 1:
                mu[k, l] = np.sum(weights * X[:, l]) / weight_sum
    
    # Update variances (σ²)
    for k in range(K):
        # Calculate squared differences for observed dimensions
        diff = X - mu[k]  # Difference between data points and component mean
        diff_obs = diff * observed  # Zero out missing dimensions
        sq_diff = np.sum(diff_obs**2, axis=1)  # Sum squared differences per point
        
        # Count observed dimensions for each data point
        n_obs_dims = np.sum(observed, axis=1)
        
        # Calculate denominator: sum of (posterior * number of observed dimensions)
        denominator = np.sum(post[:, k] * n_obs_dims)
        
        # Calculate variance with protection against division by zero
        if denominator > 0:
            var[k] = np.sum(post[:, k] * sq_diff) / denominator
        else:
            var[k] = min_variance
        
        # Apply minimum variance constraint to prevent collapse
        var[k] = max(var[k], min_variance)
    
    return GaussianMixture(mu, var, p)




def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_ll = None
    ll = None

    while True:
        # E-step
        post, ll = estep(X, mixture)
        
        # Check for convergence
        if prev_ll is not None and abs(ll - prev_ll) <= 1e-6 * abs(ll):
            break
        prev_ll = ll

        # M-step
        mixture = mstep(X, post, mixture)
    
    return mixture, post, ll



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns:
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    
    # Create a copy of X to fill in missing values
    X_pred = X.copy()
    
    # Create a mask for observed entries (1 if observed, 0 if missing)
    observed = (X != 0).astype(float)
    
    # Calculate unconditional mean (used for completely missing rows)
    # E[x] = sum_k p_k * mu_k
    unconditional_mean = np.sum(mixture.p[:, np.newaxis] * mixture.mu, axis=0)
    
    # For each data point with missing values
    for i in range(n):
        # Skip if no missing values
        if np.all(observed[i]):
            continue
        
        # Indices of observed and missing features
        obs_idx = np.where(observed[i] == 1)[0]
        miss_idx = np.where(observed[i] == 0)[0]
        
        # If no observed values, use unconditional mean
        if len(obs_idx) == 0:
            X_pred[i] = unconditional_mean
            continue
        
        # Extract observed values
        x_obs = X[i, obs_idx]
        
        # Compute posterior probabilities for each component
        post = np.zeros(K)
        for k in range(K):
            # Extract mean and variance for this component
            mu_k_obs = mixture.mu[k, obs_idx]
            var_k = mixture.var[k]
            
            # Compute log-likelihood of observed values
            diff = x_obs - mu_k_obs
            loglike = -0.5 * np.sum(diff**2) / var_k - 0.5 * len(obs_idx) * np.log(2 * np.pi * var_k)
            
            # Add log prior
            post[k] = np.log(mixture.p[k]) + loglike
        
        # Normalize posterior (using log-sum-exp trick)
        post_max = np.max(post)
        post = np.exp(post - post_max)
        post = post / np.sum(post)
        
        # Compute expected values for missing features
        for j in miss_idx:
            # E[x_j | x_obs] = sum_k P(k | x_obs) * mu_k,j
            X_pred[i, j] = np.sum(post * mixture.mu[:, j])
    
    return X_pred
