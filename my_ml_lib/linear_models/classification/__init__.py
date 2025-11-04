from ._lda import LinearDiscriminantAnalysis
from ._logistic import LogisticRegression # Added
# Optional imports
from ._least_squares import LeastSquaresClassifier
from ._perceptron import Perceptron

__all__ = [
    'LinearDiscriminantAnalysis',
    'LogisticRegression', # Added
    'LeastSquaresClassifier', # Optional
    'Perceptron'             # Optional
]