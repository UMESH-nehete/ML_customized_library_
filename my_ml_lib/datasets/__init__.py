from ._loaders import load_spambase, load_fashion_mnist,DatasetNotFoundError
from ._synthetic import make_noisy_sine

__all__ = [
    'load_spambase',
    'load_fashion_mnist',
    'make_noisy_sine',
    'DatasetNotFoundError'
]