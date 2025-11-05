# my_ml_lib/preprocessing/_gaussian.py

import numpy as np

class GaussianBasisFeatures:
    def __init__(self, n_centers=100, sigma=5.0, random_state=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers_ = None
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Select n_centers random points from X as RBF centers (mu_j).
        """
        rng = np.random.RandomState(self.random_state)
        
        # Use rng.choice for reproducible random sampling
        if X.shape[0] < self.n_centers:
            # Handle case where n_samples < n_centers
            indices = rng.choice(X.shape[0], X.shape[0], replace=False)
        else:
            indices = rng.choice(X.shape[0], self.n_centers, replace=False)
            
        self.centers_ = X[indices]
        return self

    def transform(self, X):
        """
        Transform input X into Gaussian RBF features.
        Output shape: (n_samples, n_centers)
        """
        if self.centers_ is None:
            raise RuntimeError("Transformer is not fitted yet.")

        # Compute squared Euclidean distance between each sample and each center
        # X shape: (n_samples, n_features)
        # centers_ shape: (n_centers, n_features)
        
        # 1. Expand dimensions for broadcasting (X[:, newaxis, :] - centers_[newaxis, :, :])
        # Resulting shape (n_samples, n_centers, n_features)
        diff_sq = (X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :])**2 
        
        # 2. Sum along the feature axis (axis=2) to get squared distance
        # dist_sq shape: (n_samples, n_centers)
        dist_sq = np.sum(diff_sq, axis=2)

        # 3. Apply Gaussian RBF formula: exp(- ||X - center||^2 / (2 * sigma^2))
        Phi = np.exp(- dist_sq / (2 * self.sigma**2))
        
        return Phi 
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)