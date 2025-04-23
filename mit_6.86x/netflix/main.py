import numpy as np
import kmeans
from common import init, plot, bic, rmse
import naive_em
import em
from matplotlib import pyplot as plt

X = np.loadtxt("netflix_incomplete.txt")

# K-means Clustering
k_means_results = {}

"""for K in [1, 2, 3, 4]:
    best_cost = float('inf')
    best_mixture = None
    best_post = None
    best_seed = None
    
    # Try different seeds and select the best result
    for seed in range(5):
        # Initialize mixture model with random means
        mixture, post = init(X, K, seed)
        
        # Run K-means algorithm
        mixture, post, cost = kmeans.run(X, mixture, post)
        
        # Track the best solution
        if cost < best_cost:
            best_cost = cost
            best_mixture = mixture
            best_post = post
            best_seed = seed
    
    # Store the best result
    k_means_results[K] = {
        'cost': best_cost,
        'mixture': best_mixture,
        'post': best_post,
        'seed': best_seed
    }
    
    # Plot the best solution
    title = f"K-means Clustering (K={K}, Seed={best_seed})"
    plot(X, best_mixture, best_post, title)
    
    print(f"Cost|K={K} = {best_cost:.4f}")"""

# EM Clustering
if True:
    X_incomplete = np.loadtxt("netflix_incomplete.txt")
    X_gold = np.loadtxt("netflix_complete.txt")
    
    print(f"Data loaded: {X_incomplete.shape} incomplete matrix")
    
    # Find the best mixture model for K=12
    K = 12
    best_ll = float('-inf')
    best_mixture = None
    best_post = None
    best_seed = None
    
    # Try different initializations to find the best model
    for seed in range(5):
        print(f"Testing seed {seed}...", end=" ")
        
        # Initialize mixture model parameters
        mixture, post = init(X_incomplete, K, seed)
        
        # Run EM algorithm until convergence
        mixture, post, ll = em.run(X_incomplete, mixture, post)
        
        print(f"Log-likelihood: {ll:.6f}")
        
        # Keep track of the best model
        if ll > best_ll:
            best_ll = ll
            best_mixture = mixture
            best_post = post
            best_seed = seed
    
    print(f"\nBest model: seed={best_seed}, log-likelihood={best_ll:.6f}")
    
    # Fill in the missing entries using the best mixture model
    X_pred = em.fill_matrix(X_incomplete, best_mixture)
    
    # Calculate RMSE between predictions and gold standard
    error = rmse(X_gold, X_pred)
    
    print("\nComparing with gold targets")
    print(f"RMSE: {error:.6f}")