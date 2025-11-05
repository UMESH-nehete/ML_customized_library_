# my_ml_lib/nn/autograd.py
import numpy as np

class Value:
    """
    A lightweight autograd Value that wraps numpy arrays and builds a computation graph.
    Supports broadcasting-aware backpropagation for common ops.
    """
    def __init__(self, data, _parents=(), _op='', label=''):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float64)

        self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float64)

        # Graph bookkeeping
        self._backward = lambda: None
        self._prev = set(_parents)
        self._op = _op
        self.label = label

    def __repr__(self):
        dshape = tuple(self.data.shape)
        return f"Value(shape={dshape}, op='{self._op}')"

    # ---------------- Utility ----------------
    @staticmethod
    def _unbroadcast(grad, target_shape):
        """
        Reduce `grad` (numpy array) to `target_shape` by summing over broadcasted axes.
        Works for scalar grads, lower-dim grads, and sums out leading axes.
        """
        # If target_shape is scalar
        if target_shape == () or target_shape == ():
            return np.array(grad).sum()

        grad = np.array(grad)  # ensure numpy
        # If shapes already match
        if grad.shape == target_shape:
            return grad

        # If grad is scalar -> broadcast to target
        if grad.shape == () or np.isscalar(grad):
            return np.ones(target_shape, dtype=grad.dtype) * grad

        # Add leading singleton dims to grad if necessary
        if grad.ndim < len(target_shape):
            ndiff = len(target_shape) - grad.ndim
            grad = grad.reshape((1,) * ndiff + grad.shape)

        # Now for any dimension where target_shape has 1 and grad has >1, we must sum over that axis
        for axis, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
            if t_dim == 1 and g_dim > 1:
                grad = grad.sum(axis=axis, keepdims=True)

        # If grad still has extra dims on left (shouldn't normally once reshaped), sum them
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)

        # Final check/reshape
        if grad.shape != target_shape:
            try:
                grad = grad.reshape(target_shape)
            except Exception:
                # Fallback: sum to scalar then broadcast
                grad = np.ones(target_shape, dtype=grad.dtype) * grad.sum()

        return grad

    # ---------------- Operators ----------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            g = out.grad
            # reduce to shapes
            g_self = Value._unbroadcast(g, self.data.shape)
            g_other = Value._unbroadcast(g, other.data.shape)
            self.grad += g_self
            other.grad += g_other

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            g = out.grad
            g_self = g * other.data
            g_other = g * self.data

            g_self = Value._unbroadcast(g_self, self.data.shape)
            g_other = Value._unbroadcast(g_other, other.data.shape)

            self.grad += g_self
            other.grad += g_other

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * (self ** -1)

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Only scalar powers supported"
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            # d/dx x^p = p * x^(p-1)
            g = out.grad * (power * (self.data ** (power - 1)))
            g = Value._unbroadcast(g, self.data.shape)
            self.grad += g

        out._backward = _backward
        return out

    # ---------------- Activations / elementwise ----------------
    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            mask = (self.data > 0).astype(float)
            g = out.grad * mask
            g = Value._unbroadcast(g, self.data.shape)
            self.grad += g

        out._backward = _backward
        return out

    def exp(self):
        clipped = np.clip(self.data, -500, 500)
        out = Value(np.exp(clipped), (self,), 'exp')

        def _backward():
            # derivative = exp(original)
            g = out.grad * out.data
            g = Value._unbroadcast(g, self.data.shape)
            self.grad += g

        out._backward = _backward
        return out

    def log(self):
        eps = 1e-12
        safe = np.maximum(self.data, eps)
        out = Value(np.log(safe), (self,), 'log')

        def _backward():
            g = out.grad * (1.0 / safe)
            g = Value._unbroadcast(g, self.data.shape)
            self.grad += g

        out._backward = _backward
        return out

    # ---------------- MatMul ----------------
    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            A, B = self.data, other.data
            G = out.grad  # shape matches A@B

            # special case: 1D vectors dot -> scalar
            if A.ndim == 1 and B.ndim == 1:
                # out is scalar
                self.grad += G * B
                other.grad += G * A
                return

            # General: A(n,d) @ B(d,m) -> out(n,m)
            grad_A = G @ B.T
            grad_B = A.T @ G

            grad_A = Value._unbroadcast(grad_A, A.shape)
            grad_B = Value._unbroadcast(grad_B, B.shape)

            self.grad += grad_A
            other.grad += grad_B

        out._backward = _backward
        return out

    # ---------------- Reductions ----------------
    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Value(np.array(out_data), (self,), 'sum')

        def _backward():
            g = out.grad
            # broadcast g back: unbroadcast will take care
            grad_b = Value._unbroadcast(g, self.data.shape)
            self.grad += grad_b

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Value(np.array(out_data), (self,), 'mean')

        def _backward():
            # number of elements averaged over
            if axis is None:
                N = self.data.size
            else:
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = tuple(axis)
                N = 1
                for a in axes:
                    N *= self.data.shape[a]
            g = out.grad / N
            grad_b = Value._unbroadcast(g, self.data.shape)
            self.grad += grad_b

        out._backward = _backward
        return out

    # ---------------- Backpropagation ----------------
    def backward(self):
        """
        Build topological order and call local backward functions in reverse order.
        """
        # initialize grad of root to ones (same shape as data)
        self.grad = np.ones_like(self.data, dtype=np.float64)

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)

        build(self)

        for node in reversed(topo):
            node._backward()
# ... [Existing Value class code, including the backward method] ...

def get_all_nodes_and_edges(root_node):
    """
    Traverses the computation graph backward from the root node to collect 
    all unique Value nodes and the edges (connections) between them.
    Used for graph visualization (Problem 4.4a).
    """
    nodes = set()
    edges = set()

    def build_sets(node):
        nodes.add(node)
        # _prev is the set of parents (children in the forward sense)
        for child in node._prev:
            # Edge is represented as (parent_node_id, current_node_id)
            # The edge connects the parent (child._prev) to the result (node)
            edges.add((child, node)) 
            build_sets(child)

    build_sets(root_node)
    
    return nodes, edges