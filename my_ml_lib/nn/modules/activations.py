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
        # sigmoid = 1 / (1 + exp(-x))
        one = Value(np.array(1.0))
        return one / (one + (-x).exp())

    def __repr__(self):
        return "Sigmoid()"

    def parameters(self):
        return iter([])
