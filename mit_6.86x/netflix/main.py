import numpy as np
import kmeans
from common import init, plot
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# K-means Clustering
k_means_results = {}

for K in [1, 2, 3, 4]:
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
    
    print(f"Cost|K={K} = {best_cost:.4f}")
