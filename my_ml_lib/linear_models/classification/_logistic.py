# my_ml_lib/linear_models/classification/_logistic.py
import numpy as np

class LogisticRegression:
    def __init__(self, alpha=0.0, lr=0.01, max_iter=100, fit_intercept=True):
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.w_ = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _augment(self, X):
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0],1)), X])
        return X

    def fit(self, X, y):
        X_aug = self._augment(X)
        n, d = X_aug.shape

        self.w_ = np.zeros(d)

        for epoch in range(self.max_iter):
            h = self._sigmoid(X_aug @ self.w_)
            grad = X_aug.T @ (h - y) / n

        # regularization except bias
            reg = self.alpha * np.r_[0, self.w_[1:]] if self.fit_intercept else self.alpha*self.w_

            self.w_ -= self.lr * (grad + reg)

        return self

    def predict_proba(self, X):
        X_aug = self._augment(X)
        p1 = self._sigmoid(X_aug @ self.w_)
        return np.column_stack([1-p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)[:,1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
