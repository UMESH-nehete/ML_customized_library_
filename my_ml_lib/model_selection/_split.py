import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split X and y arrays into random train and test subsets.
    """
    n_samples = X.shape[0]
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # ---- Step 1 - compute n_test & n_train ----
    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size float must be between 0 and 1")
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise ValueError("test_size must be float or int")

    n_train = n_samples - n_test

    # ---- Step 2 - create + shuffle indices ----
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    # ---- Step 3 - split ----
    test_indices  = indices[:n_test]
    train_indices = indices[n_test:]

    # ---- Step 4 - slice arrays ----
    X_train = X[train_indices]
    X_test  = X[test_indices]
    y_train = y[train_indices]
    y_test  = y[test_indices]

    return X_train, X_test, y_train, y_test
