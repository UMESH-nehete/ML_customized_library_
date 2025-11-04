import numpy as np

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError("n_splits must be an integer > 1")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
            
        # fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start = current
            end   = current + fold_size
          
            test_indices  = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))

            yield train_indices, test_indices

            current = end

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
