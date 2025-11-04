# In my_ml_lib/__init__.py

"""
my_ml_lib: A foundational machine learning library.

This library provides implementations of common machine learning algorithms
and tools, built primarily using NumPy, developed as part of the
Foundations of Machine Learning course assignments.
"""

# Version of the library
__version__ = "0.1.0"

# List of publicly available modules (subpackages)
_submodules = [
    "datasets",
    "linear_models",
    "model_selection",
    "naive_bayes",
    "preprocessing",
    "nn",
]

# Define what gets imported with 'from my_ml_lib import *'
# Primarily includes the submodules. You could add specific important
# classes/functions here if desired, but listing modules is common.
__all__ = _submodules

