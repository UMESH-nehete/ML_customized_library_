import numpy as np
# May need itertools.combinations_with_replacement
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        # Add any other attributes needed to store fitted info if necessary

    def fit(self, X, y=None):
        # Usually fit doesn't do much here unless you need input dimensions
        self.n_input_features_ = X.shape[1]
        return self

    def transform(self, X):
        
        # TODO: Generate polynomial features up to self.degree
        # Remember to handle include_bias
        print("PolynomialFeatures: Transform method needs implementation.")
        # Example for degree 2, include_bias=True, X = [a, b]
        # Output should be [1, a, b, a^2, ab, b^2]
        # Consider using sklearn's implementation as a reference for a robust solution
#---     
        if not hasattr(self, "n_input_features_"):
            raise RuntimeError("PolynomialFeatures not fitted. Call fit first.")

        # start with bias if included
        features = []

        if self.include_bias:
            features.append(np.ones((X.shape[0], 1)))

        # for each degree d = 1 .. D
        for d in range(1, self.degree + 1):
            for comb in combinations_with_replacement(range(self.n_input_features_), d):
                f = np.prod(X[:, comb], axis=1, keepdims=True)
                features.append(f)

        return np.hstack(features)
#---
  
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)