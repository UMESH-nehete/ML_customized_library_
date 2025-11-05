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
    def __init__(self):
        super().__init__()

    def __call__(self, x: Value) -> Value:
        # Implemented using Value operations
        # 1. Compute -z
        neg_x = -x 
        
        # 2. Compute exp(-z)
        exp_neg_x = neg_x.exp()
        
        # 3. Compute 1 + exp(-z)
        denominator = 1.0 + exp_neg_x
        
        # 4. Compute 1 / denominator
        out = denominator ** -1
        
        return out

    def __repr__(self):
        return "Sigmoid()"

    def parameters(self):
        return iter([])
