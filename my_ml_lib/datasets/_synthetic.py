import numpy as np

def make_noisy_sine(n_samples=100, noise=0.1, random_state=None):
    """
    Generates the noisy sine wave dataset from A2.

    Args:
        n_samples (int): Number of data points to generate.
        noise (float): Standard deviation of the Gaussian noise added to y.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        tuple: (X, y) numpy arrays. X is (n_samples, 1), y is (n_samples,).


    """
#---
    if random_state is not None:
        np.random.seed(random_state)
    
    X=np.random.rand(n_samples,1)
    y=np.sin(2*np.pi*X[:,0])

    y= y_true+ np.random.normal(0,noise,size=n_samples)
    return X, y
#---
