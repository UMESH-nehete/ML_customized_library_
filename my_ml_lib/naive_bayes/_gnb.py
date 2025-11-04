import numpy as np

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes (GNB) classifier.

    Assumes features are conditionally independent given the class,
    and each feature follows a Gaussian distribution within each class.
    """
    def __init__(self):
        self.classes_ = None
        self.class_priors_ = None # P(y=k)
        self.theta_ = None # Mean of each feature per class, shape (n_classes, n_features)
        self.var_ = None   # Variance of each feature per class, shape (n_classes, n_features)
        self.epsilon_ = 1e-9 # To prevent division by zero in variance

    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes according to X, y.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
        """
        # TODO: Implement GNB fitting logic from A2.
        # 1. Find unique classes: self.classes_ = np.unique(y)
        # 2. Calculate class priors: self.class_priors_
        # 3. Calculate mean (self.theta_) and variance (self.var_) for each feature in each class.
        #    Remember to add self.epsilon_ to the variance.
        print("GaussianNaiveBayes: Fit method needs implementation.")
        self.classes_ = np.unique(y) if y is not None else np.array([])
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.class_priors_ = np.zeros(n_classes) # Placeholder
        self.theta_ = np.zeros((n_classes, n_features)) # Placeholder
        self.var_ = np.ones((n_classes, n_features)) # Placeholder
        return self

    def _gaussian_log_pdf(self, X, class_idx):
        """Calculate log probability density function for Gaussian."""
        mean = self.theta_[class_idx]
        var = self.var_[class_idx]
        # TODO: Implement Gaussian log PDF calculation
        # log P(x_j | y=k) = -0.5 * log(2*pi*var_j) - 0.5 * ((X_j - mean_j)^2 / var_j)
        print("GNB: _gaussian_log_pdf needs implementation.")
        # log_prob = -0.5 * np.sum(np.log(2. * np.pi * var)) \
        #            - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
        # return log_prob # Shape should be (n_samples,)
        return np.zeros(X.shape[0]) # Placeholder


    def predict_log_proba(self, X):
        """
        Calculate log probability estimates for samples in X.

    'PolynomialF
        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Log probability of samples for each class, shape (n_samples, n_classes).
        """
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # TODO: Implement log probability prediction.
        # For each class k:
        #   log P(y=k | x) proportional to log P(x | y=k) + log P(y=k)
        #   log P(x | y=k) = sum over features j [ log P(x_j | y=k) ] (due to naive assumption)
        #   Use _gaussian_log_pdf for each feature's contribution.
        print("GaussianNaiveBayes: predict_log_proba needs implementation.")
        # joint_log_likelihood = np.zeros((X.shape[0], len(self.classes_)))
        # for i, k in enumerate(self.classes_):
        #    # Calculate sum of log PDFs for all features for class k
        #    # Need careful implementation of _gaussian_log_pdf to work feature-wise or adjust here
        #    class_conditional_log_prob = self._gaussian_log_pdf(X, i) # Placeholder logic
        #    joint_log_likelihood[:, i] = np.log(self.class_priors_[i]) + class_conditional_log_prob

        # return joint_log_likelihood
        return np.zeros((X.shape[0], len(self.classes_))) # Placeholder

    def predict_proba(self, X):
        """
        Calculate probability estimates for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability of samples for each class, shape (n_samples, n_classes).
        """
        log_proba = self.predict_log_proba(X)
        # Convert log probabilities to probabilities (handle numerical stability)
        # TODO: Implement stable softmax or equivalent normalization
        print("GaussianNaiveBayes: predict_proba needs implementation.")
        # Example stable softmax:
        # max_log_proba = np.max(log_proba, axis=1, keepdims=True)
        # proba = np.exp(log_proba - max_log_proba)
        # proba /= np.sum(proba, axis=1, keepdims=True)
        # return proba
        proba = np.exp(log_proba) / np.sum(np.exp(log_proba), axis=1, keepdims=True) # Naive exp
        return proba # Placeholder


    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        log_proba = self.predict_log_proba(X)
        # Find the class with the highest log probability
        # TODO: Implement argmax logic
        print("GaussianNaiveBayes: Predict method needs implementation.")
        # return self.classes_[np.argmax(log_proba, axis=1)]
        return np.zeros(X.shape[0]) # Placeholder