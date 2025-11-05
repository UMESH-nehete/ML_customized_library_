# my_ml_lib/nn/modules/activations.py
from .base import Module
from ..autograd import Value
import numpy as np

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Value) -> Value:
        return x.relu()

    def __repr__(self):
        return "ReLU()"

    def parameters(self):
        # no parameters
        return iter([])

class Sigmoid(Module):
    """
    Applies the sigmoid function: sigma(z) = 1 / (1 + exp(-z)) (Problem 4.2).
    """
    def __call__(self, x: Value) -> Value:
        # Implemented using Value operations
        # 1. Compute -z
        neg_x = -x 
        
        # 2. Compute exp(-z)
        exp_neg_x = neg_x.exp()
        
        # 3. Compute 1 + exp(-z)
        denominator = 1.0 + exp_neg_x # Constant 1.0 is handled by Value.__add__
        
        # 4. Compute 1 / denominator (using power -1)
        out = denominator ** -1 
        
        return out
    
    def __repr__(self):
        return "Sigmoid()"

    def parameters(self):
        return iter([])
