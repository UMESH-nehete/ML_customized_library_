import numpy as np

class Perceptron:
    """
    Perceptron classifier.

    Simple algorithm for binary linear classification.
    Uses iterative updates based on misclassified points.
    """
    def __init__(self, learning_rate=0.01, max_iters=1000, random_state=None):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.random_state = random_state
        self.w_ = None # Weights (including bias)
        self.errors_ = [] # To store number of misclassifications per epoch

    def fit(self, X, y):
        """
        Fit perceptron model.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels, e.g., 0/1 or -1/1), shape (n_samples,).
        """
        # TODO: Implement the Perceptron learning algorithm (Pocket algorithm recommended).
        # 1. Initialize weights (e.g., randomly or to zeros). Include bias.
        # 2. Augment X with bias column.
        # 3. Ensure y labels are appropriate (+1/-1 often used).
        # 4. Iterate up to max_iters:
        #    - Loop through samples (or shuffled samples).
        #    - Make a prediction using the current weights.
        #    - If misclassified, update weights: w = w + learning_rate * (y_true - y_pred) * x
        #    - Optionally, implement Pocket algorithm: keep track of the weights
        #      that achieved the lowest misclassification rate so far.
        print("Perceptron: Fit method needs implementation.")
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # Placeholder init
        self.errors_ = []
        return self

    def _predict_raw(self, X):
        """Calculate net input (scores)."""
        # Augment X
        X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_augmented @ self.w_

    def predict(self, X):
        """
        Return class label after unit step.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (e.g., 0/1 or -1/1 depending on fit).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # TODO: Apply activation function (step function) to raw predictions.
        # Map output to the format used during training (e.g., 0/1 or -1/1).
        print("Perceptron: Predict method needs implementation.")
        # scores = self._predict_raw(X)
        # predictions = np.where(scores >= 0.0, 1, -1) # Example for -1/1 labels
        # return predictions
        return np.zeros(X.shape[0]) # Placeholder