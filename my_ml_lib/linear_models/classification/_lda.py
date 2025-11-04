# import numpy as np

# class LinearDiscriminantAnalysis:
#     """
#     Linear Discriminant Analysis (LDA) classifier.

#     Assumes class conditional densities are Gaussians with shared covariance.
#     Uses the least-squares approach as implemented in A2.
#     """
#     def __init__(self):
#         self.w_ = None # Projection vector (including bias term)

#     def fit(self, X, y):
#         """
#         Fit the LDA model according to the given training data.

#         Args:
#             X (np.ndarray): Training vectors, shape (n_samples, n_features).
#             y (np.ndarray): Target values (class labels), shape (n_samples,).
#                             Assumes binary classification with labels like 0 and 1.
#         """
#         # TODO: Implement the LDA fitting logic from A2.
#         # 1. Estimate class priors pi_k
#         # 2. Estimate class means mu_k
#         # 3. Estimate the shared covariance matrix Sigma
#         # 4. Calculate the weight vector w and bias w_0 (or augment X and find w_)
#         #    using the formulas derived from Gaussian distributions or the
#         #    least-squares approach if that was used in A2.
#         # Example (Least Squares Approach from A2):
#         # - Augment X with a column of ones: X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])
#         # - Convert y to a suitable target representation T (e.g., using one-hot encoding
#         #   or specific target values as discussed in Bishop for least squares classification).
#         # - Calculate weights: self.w_ = np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ T
#         #   (Need to be careful about the multi-class least squares formulation if applicable,
#         #   though the prompt implies binary LDA for simplicity here).
#         #   For binary LDA derived from Gaussian assumptions, the calculation is different.
#         print("LDA: Fit method needs implementation based on A2 logic.")
#         self.w_ = np.zeros(X.shape[1] + 1) # Placeholder (includes bias)
#         return self

#     def predict(self, X):
#         """
#         Predict class labels for samples in X.

#         Args:
#             X (np.ndarray): Samples, shape (n_samples, n_features).

#         Returns:
#             np.ndarray: Predicted class labels, shape (n_samples,).
#         """
#         if self.w_ is None:
#             raise RuntimeError("Model is not fitted yet.")

#         # TODO: Implement prediction logic.
#         # 1. Augment X with a column of ones.
#         # 2. Calculate the linear score: scores = X_augmented @ self.w_
#         # 3. Apply a threshold (typically 0) to determine the class.
#         #    Need to map the output to the original class labels (e.g., 0 and 1).
#         print("LDA: Predict method needs implementation.")
#         # Example:
#         # X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])
#         # scores = X_augmented @ self.w_
#         # predictions = (scores > 0).astype(int) # Assumes mapping to 0/1
#         # return predictions
#         return np.zeros(X.shape[0]) # Placeholder





import numpy as np

class LinearDiscriminantAnalysis:
    """
    Linear Discriminant Analysis (LDA) classifier using the least squares approach.

    Fits a linear regression model to predict target values derived from class labels.
    """
    def __init__(self):
        self.w_ = None # Projection vector (including bias term)

    def fit(self, X, y):
        """
        Fit the LDA model using the least squares target encoding.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels, 0 or 1), shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("This LDA implementation currently supports only binary classification.")
        
        # --- Augment X with a bias column ---
        X_augmented = np.hstack([np.ones((n_samples, 1)), X])

        # --- Create target vector T based on class membership ---
        # Target for class 1: N / N_1
        # Target for class 0: -N / N_0
        N = n_samples
        N_1 = np.sum(y == self.classes_[1]) # Count of class 1
        N_0 = np.sum(y == self.classes_[0]) # Count of class 0

        # Ensure we don't divide by zero if a class is empty (unlikely but safe)
        if N_1 == 0 or N_0 == 0:
            raise ValueError("Training data must contain samples from both classes.")

        T = np.where(y == self.classes_[1], N / N_1, -N / N_0)

        # --- Calculate weights using pseudo-inverse ---
        # w = (X_aug^T X_aug)^(-1) X_aug^T T  =>  w = pinv(X_aug) @ T
        try:
            # Use pseudo-inverse for numerical stability
            self.w_ = np.linalg.pinv(X_augmented) @ T
            # Alternative: explicit normal equation (less stable if X^T X is singular)
            # self.w_ = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ T
        except np.linalg.LinAlgError:
            print("Error: Could not compute weights using pseudo-inverse. Matrix might be ill-conditioned.")
            self.w_ = None # Indicate fitting failed
            raise # Re-raise the error

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1), shape (n_samples,).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # --- Augment X with a bias column ---
        X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])

        # --- Calculate the linear scores ---
        scores = X_augmented @ self.w_

        # --- Apply threshold and map to original class labels ---
        # Threshold is typically 0 for this target encoding
        # Assign to class 1 if score > 0, otherwise class 0
        predictions = np.where(scores > 0, self.classes_[1], self.classes_[0])

        return predictions

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)