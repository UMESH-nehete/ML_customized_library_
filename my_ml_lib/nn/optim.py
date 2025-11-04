# my_ml_lib/nn/optim.py
import numpy as np
from .autograd import Value

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
        for p in self.params:
            if not isinstance(p, Value):
                raise TypeError("Optimizer parameters must be Value objects.")
            if not hasattr(p, 'grad') or p.grad is None:
                p.grad = np.zeros_like(p.data, dtype=np.float64)

    def step(self):
        """
        Performs an in-place SGD update: p.data = p.data - lr * p.grad
        """
        for p in self.params:
            if p is None:
                continue
            # in-place update (preferred)
            p.data[:] = p.data - self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p is None:
                continue
            p.grad = np.zeros_like(p.data, dtype=np.float64)

    def __repr__(self):
        return f"SGD(lr={self.lr})"
