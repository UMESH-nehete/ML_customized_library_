# my_ml_lib/preprocessing/_polynomial.py

import numpy as np

class PolynomialFeatures:
    """
    Generates polynomial features up to a specified degree.
    Includes interaction terms.
    """
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self.n_output_features_ = None
        self.n_features_in_ = None
        self._powers = None 

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        
        # The key is determining all combinations of feature powers that sum up to 'degree'
        # This implementation uses a recursive helper function to build the power matrix.
        
        from itertools import combinations_with_replacement
        
        if self.n_features_in_ == 0:
            self._powers = np.empty((0, 0), dtype=np.int32)
        else:
            # Generate all combinations of feature indices (0 to n_features_in_ - 1)
            # where the length of the combination is up to 'degree'
            
            # Start with an empty list for the final power matrix
            self._powers = []
            
            # Iterate through all degrees from 1 up to the max degree
            for d in range(1, self.degree + 1):
                # Combinations_with_replacement generates index tuples (j1, j2, ...)
                for combo in combinations_with_replacement(range(self.n_features_in_), d):
                    # Convert the index combo into a power vector
                    power_vector = np.zeros(self.n_features_in_, dtype=int)
                    for index in combo:
                        power_vector[index] += 1
                    self._powers.append(power_vector)

            # Convert to numpy array
            self._powers = np.array(self._powers, dtype=np.int32)

        # Count output features (including the bias term)
        self.n_output_features_ = len(self._powers) + (1 if self.include_bias else 0)
        
        return self

    def transform(self, X):
        """Transforms X using the powers calculated in fit."""
        if self._powers is None:
            raise RuntimeError("PolynomialFeatures must be fitted before calling transform.")
        
        n_samples = X.shape[0]
        
        # Initialize the output matrix with the bias column if needed
        if self.include_bias:
            X_poly = np.ones((n_samples, self.n_output_features_))
            current_feature_index = 1
        else:
            X_poly = np.zeros((n_samples, self.n_output_features_))
            current_feature_index = 0
            
        # Iterate over each power vector (column) in the output
        for power_vector in self._powers:
            # Calculate the new feature by raising input features to their respective power
            # The product is taken across all features (axis=1)
            new_feature = np.prod(X ** power_vector, axis=1)
            X_poly[:, current_feature_index] = new_feature
            current_feature_index += 1
            
        return X_poly

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)