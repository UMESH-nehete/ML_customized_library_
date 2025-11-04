import numpy as np

class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes (BNB) classifier.

    Assumes features are binary (0 or 1), conditionally independent
    given the class, and follow a Bernoulli distribution within each class.
    Often used for text classification with presence/absence features.
    """
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Additive (Laplace/Lidstone) smoothing parameter
                           (0 for no smoothing). Corresponds to Beta(alpha, alpha) prior.
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None # Log P(y=k)
        self.feature_log_prob_ = None # Log P(x_j=1 | y=k), shape (n_classes, n_features)
        # We implicitly store log P(x_j=0 | y=k) as log(1 - exp(log P(x_j=1 | y=k)))

    def fit(self, X, y):
        """
        Fit Bernoulli Naive Bayes according to X, y.

        Assumes X contains binary features (0 or 1).

        Args:
            X (np.ndarray): Training vectors (binary), shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
        """
        # TODO: Implement BNB fitting logic from A2.
        # 1. Find unique classes: self.classes_
        # 2. Calculate class log priors: self.class_log_prior_
        # 3. For each class k and feature j, estimate P(x_j=1 | y=k).
        #    - Count occurrences N_k = count(y=k)
        #    - Count occurrences N_kj = count(x_j=1 and y=k)
        #    - Apply Laplace smoothing: P(x_j=1 | y=k) = (N_kj + alpha) / (N_k + 2*alpha)
        # 4. Store the log of these probabilities: self.feature_log_prob_
        print("BernoulliNaiveBayes: Fit method needs implementation.")
        self.classes_ = np.unique(y) if y is not None else np.array([])
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.class_log_prior_ = np.zeros(n_classes) # Placeholder
        self.feature_log_prob_ = np.zeros((n_classes, n_features)) # Placeholder
        return self

    def predict_log_proba(self, X):
        """
        Calculate log probability estimates for samples in X.

        Assumes X contains binary features (0 or 1).

        Args:
            X (np.ndarray): Samples (binary), shape (n_samples, n_features).

        Returns:
            np.ndarray: Log probability of samples for each class, shape (n_samples, n_classes).
        """
        if self.classes_ is None:
             raise RuntimeError("Model is not fitted yet.")

        # TODO: Implement log probability prediction for Bernoulli features.
        # For each class k:
        #   log P(y=k | x) proportional to log P(y=k) + sum over features j [ log P(x_j | y=k) ]
        #   log P(x_j | y=k) = x_j * log P(x_j=1|y=k) + (1-x_j) * log P(x_j=0|y=k)
        #                 = x_j * log P(x_j=1|y=k) + (1-x_j) * log(1 - P(x_j=1|y=k))
        # Use self.feature_log_prob_ and calculate log(1-exp(feature_log_prob_)) carefully.
        print("BernoulliNaiveBayes: predict_log_proba needs implementation.")
        # Example calculation using log probabilities directly:
        # log_prob_x1 = self.feature_log_prob_ # Log P(xj=1|yk)
        # log_prob_x0 = np.log(1 - np.exp(log_prob_x1)) # Log P(xj=0|yk)
        # joint_log_likelihood = np.zeros((X.shape[0], len(self.classes_)))
        # for i, k in enumerate(self.classes_):
        #     # Calculate sum using broadcasting and matrix multiplication:
        #     # log P(x|yk) = X @ log_prob_x1[i].T + (1-X) @ log_prob_x0[i].T
        #     # Be careful with dimensions here. A simple sum might be easier:
        #     log_p_x_given_k = X * log_prob_x1[i] + (1 - X) * log_prob_x0[i]
        #     sum_log_p = np.sum(log_p_x_given_k, axis=1)
        #     joint_log_likelihood[:, i] = self.class_log_prior_[i] + sum_log_p
        # return joint_log_likelihood
        return np.zeros((X.shape[0], len(self.classes_))) # Placeholder

    def predict_proba(self, X):
        """
        Calculate probability estimates for samples in X.

        Args:
            X (np.ndarray): Samples (binary), shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability of samples for each class, shape (n_samples, n_classes).
        """
        log_proba = self.predict_log_proba(X)
        # Convert log probabilities to probabilities (handle numerical stability)
        # TODO: Implement stable softmax or equivalent normalization
        print("BernoulliNaiveBayes: predict_proba needs implementation.")
        proba = np.exp(log_proba) / np.sum(np.exp(log_proba), axis=1, keepdims=True) # Naive exp
        return proba # Placeholder


    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (np.ndarray): Samples (binary), shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        log_proba = self.predict_log_proba(X)
        # Find the class with the highest log probability
        # TODO: Implement argmax logic
        print("BernoulliNaiveBayes: Predict method needs implementation.")
        # return self.classes_[np.argmax(log_proba, axis=1)]
        return np.zeros(X.shape[0]) # Placeholder