


    # In my_ml_lib/preprocessing/_data.py
import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None # Will store standard deviation


    ## ToDO
    def fit(self, X, y=None):
        """
        Compute the mean and standard deviation to be used for later scaling.

        Args:
            X (np.ndarray): The data used to compute the mean and standard deviation,
                            shape (n_samples, n_features).
            y (None): Ignored. Present for API consistency.
        """
        # Calculate mean and std deviation along features (axis=0)
    
        
        # Handle features with zero standard deviation (avoid division by zero)
        # If std dev is 0, scaling won't change the value (it's already 0 after mean subtraction)
        # We replace scale_ with 1 in these cases to avoid NaN results.
      #---

        self.mean_=np.mean(X, axis=0)
        self.scale_=np.std(X, axis=0)
        self.scale_[self.scale_==0]=1e-8
            #---
        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.

        Args:
            X (np.ndarray): The data to scale, shape (n_samples, n_features).

        Returns:
            np.ndarray: The scaled data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
            
 
        # Apply the transformation: 
 #---
        X_scaled = (X - self.mean_) / self.scale_
 #---
        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): The data to fit and transform, shape (n_samples, n_features).
            y (None): Ignored.

        Returns:
            np.ndarray: The transformed data.
        """
        # Calls fit() and then transform()
        self.fit(X, y)
        return self.transform(X)

