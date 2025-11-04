# Import key components from submodules for easier access like nn.Linear
from .autograd import Value
from .modules import Module, Linear, ReLU, Sigmoid, Sequential
from . import optim
from . import losses
from .losses import CrossEntropyLoss # <-- CHANGE HERE: Import the class directly
__all__ = [
    'Value',
    'Module',
    'Linear',
    'ReLU',
    'Sigmoid',
    'Sequential',
    'optim',
    'losses'
]