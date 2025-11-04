# my_ml_lib/nn/modules/linear.py
import numpy as np
from .base import Module
from my_ml_lib.nn.autograd import Value

class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xW + b
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(2.0 / max(1, in_features))
        self.weight = Value(scale * np.random.randn(in_features, out_features), label='weight')

        if bias:
            self.bias = Value(np.zeros(out_features), label='bias')
        else:
            self.register_parameter('bias', None)

    def __call__(self, x: Value) -> Value:
        """
        Forward pass: x shape (batch_size, in_features) or (in_features,) for single sample.
        Weight shape: (in_features, out_features)
        Result shape: (batch_size, out_features) or (out_features,)
        """
        # matrix multiplication: x @ W
        out = x @ self.weight

        # add bias if present (broadcasting handled by Value.__add__)
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        has_bias = self._parameters.get('bias', None) is not None
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={has_bias})"
