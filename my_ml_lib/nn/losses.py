# my_ml_lib/nn/losses.py
import numpy as np
from .modules.base import Module
from .autograd import Value

class BinaryCrossEntropyLoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}.")
        self.reduction = reduction

    def __call__(self, logits: Value, targets: np.ndarray) -> Value:
        # logits: Value shape (batch,) or (batch,1)
        # targets: numpy array shape (batch,) with 0/1
        targets_np = targets.astype(np.float64).reshape(-1, 1)
        targets_v = Value(targets_np)

        # sigmoid probabilities
        probs = Value(np.array(1.0)) / (Value(np.array(1.0)) + (-logits).exp())

        # clip handled in Value.log; but we avoid direct log of zero by using small eps in Value.log
        eps = 1e-12
        # BCE per element: -[ y*log(p) + (1-y)*log(1-p) ]
        term1 = targets_v * probs.log()
        term2 = (Value(np.array(1.0)) - targets_v) * (Value(np.array(1.0)) - probs).log()
        loss_elements = -(term1 + term2)  # shape (batch, 1)
        # reduce
        if self.reduction == 'mean':
            return loss_elements.mean()
        elif self.reduction == 'sum':
            return loss_elements.sum()
        else:
            return loss_elements

class CrossEntropyLoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}.")
        self.reduction = reduction

    def __call__(self, input_logits: Value, target: np.ndarray) -> Value:
        # input_logits: Value shape (batch_size, n_classes)
        batch_size, n_classes = input_logits.data.shape

        # stable log-softmax
        # compute max per row (numpy), wrap as Value so subtraction builds graph
        max_per_row = np.max(input_logits.data, axis=1, keepdims=True)
        max_v = Value(max_per_row)
        shifted = input_logits - max_v            # Value
        exp_shift = shifted.exp()                # Value
        denom = exp_shift.sum(axis=1, keepdims=True)   # Value shape (batch,1)
        log_probs = shifted - denom.log()        # Value shape (batch, n_classes)

        # one-hot targets (numpy) then wrap as Value
        y_one_hot_np = np.zeros((batch_size, n_classes), dtype=np.float64)
        y_one_hot_np[np.arange(batch_size), target] = 1.0
        y_one_hot = Value(y_one_hot_np)

        # negative log-likelihood per sample
        # multiply and sum across classes
        nll_per_sample = - (log_probs * y_one_hot).sum(axis=1)  # Value shape (batch,)
        if self.reduction == 'mean':
            return nll_per_sample.mean()
        elif self.reduction == 'sum':
            return nll_per_sample.sum()
        else:
            return nll_per_sample

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}')"
