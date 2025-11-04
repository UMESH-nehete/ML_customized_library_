import numpy as np

class RidgeRegression:
    """
    Linear least squares with L2 regularization (Ridge Regression).

    Minimizes objective function: ||y - Xw||^2 + alpha * ||w||^2
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        """
        Args:
            alpha (float): Regularization strength; must be a positive float.
                           Larger values specify stronger regularization.
            fit_intercept (bool): Whether to calculate the intercept for this model.
                                  If set to False, no intercept will be used.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None # Weights for the features
        self.intercept_ = None # Intercept (bias term)

    def fit(self, X, y):
        """
        Fit Ridge regression model.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values, shape (n_samples,).
        """
        # TODO: Implement Ridge Regression fitting using the normal equation.
        # 1. Handle fit_intercept: If True, augment X with a column of ones.
        # 2. Create the identity matrix I (size depends on whether intercept is included).
        # 3. Important: Do not regularize the intercept term. Set the top-left element
        #    of the penalty matrix (alpha * I) to 0 if fitting intercept.
        # 4. Solve the normal equation: w = (X^T X + alpha*I)^(-1) X^T y
        print("RidgeRegression: Fit method needs implementation based on A2 normal equations.")
        n_features = X.shape[1]
        if self.fit_intercept:
            # Placeholder weights including intercept
            _w = np.zeros(n_features + 1)
            self.intercept_ = _w[0]
            self.coef_ = _w[1:]
        else:
            # Placeholder weights without intercept
            self.coef_ = np.zeros(n_features)
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        """
        Predict using the linear model.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values, shape (n_samples,).
        """
        if self.coef_ is None:
             raise RuntimeError("Model is not fitted yet.")

        # TODO: Implement prediction: y_pred = X @ coef_ + intercept_
        print("RidgeRegression: Predict method needs implementation.")
        # y_pred = X @ self.coef_ + self.intercept_
        # return y_pred
        return np.zeros(X.shape[0]) # Placeholder