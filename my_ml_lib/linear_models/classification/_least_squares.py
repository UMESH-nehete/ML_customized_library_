import numpy as np

class LeastSquaresClassifier:
    """
    Classifier using the least squares approach.

    Fits a linear model by minimizing the squared error between
    predictions and target labels (e.g., encoded as +1/-1 or one-hot).
    Prediction is typically based on the sign or argmax of the output.
    """
    def __init__(self):
        self.w_ = None # Weight vector (including bias)

    def fit(self, X, y):
        """
        Fit the least squares classifier model.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
        """
        # TODO: Implement the least squares fitting logic (from A1/A2 if covered).
        # 1. Choose a target encoding T for the labels y (e.g., +1/-1 or one-hot).
        # 2. Augment X with a bias column.
        # 3. Solve for weights using the normal equation: w = (X^T X)^(-1) X^T T
        #    Or use np.linalg.pinv for stability: w = pinv(X) @ T
        print("LeastSquaresClassifier: Fit method needs implementation.")
        self.w_ = np.zeros(X.shape[1] + 1) # Placeholder
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # TODO: Implement prediction logic.
        # 1. Augment X with a bias column.
        # 2. Calculate scores: scores = X_augmented @ self.w_
        # 3. Determine predicted class based on scores (e.g., sign for binary, argmax for multi-class)
        # 4. Map back to original class labels.
        print("LeastSquaresClassifier: Predict method needs implementation.")
        return np.zeros(X.shape[0]) # Placeholder