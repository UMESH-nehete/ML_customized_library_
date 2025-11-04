import numpy as np

class GaussianBasisFeatures:
    def __init__(self, n_centers=100, sigma=5.0, random_state=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers_ = None
        self.random_state = random_state # For reproducibility if using random sampling

    def fit(self, X, y=None):
        """
        Select n_centers random points from X as RBF centers.
        """
        # TODO: Determine the centers (mu_j).
        # Strategy: Randomly sample n_centers points from X
        # print("GaussianBasisFeatures: Fit method needs implementation.")
        rng = np.random.RandomState(self.random_state)
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
        # TODO: Apply the RBF formula: exp(-(||X - center||^2 / (2 * sigma^2)))
      
    
        # Compute squared Euclidean distance between each sample and each center
        # Using broadcasting: ||x - mu||^2 = (x - mu)^2 summed over features
        # X shape: (n_samples, n_features)
        # centers_ shape: (n_centers, n_features)
        print("GaussianBasisFeatures: Transform method needs implementation.")

        dist_sq = np.sum((X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :])**2, axis=2)

        # Gaussian RBF formula
        Phi = np.exp(- dist_sq / (2 * self.sigma**2))

        return Phi 
    
    
    
    # Placeholder - need matrix of RBF values

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

