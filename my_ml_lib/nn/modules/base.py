# my_ml_lib/nn/modules/base.py
from my_ml_lib.nn.autograd import Value # Import Value from our autograd engine
import numpy as np
from collections import OrderedDict # Use OrderedDict

class Module:
    """Base class for all neural network modules (layers, containers, etc.)."""

    def __init__(self):
        """Initializes internal dictionaries for parameters and submodules."""
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def register_parameter(self, name: str, param):
        """Adds a parameter (Value object) to the module."""
        if not isinstance(param, Value) and param is not None:
             raise TypeError(f"cannot assign {type(param)} as parameter '{name}' "
                             "(Value or None expected)")
        if '.' in name: raise KeyError("parameter name can't contain \".\"")
        if name == '': raise KeyError("parameter name can't be empty string \"\"")
        self._parameters[name] = param

    def add_module(self, name: str, module):
        """Adds a child module (another Module object) to the current module."""
        if not isinstance(module, Module) and module is not None:
             raise TypeError(f"{module} is not a Module subclass")
        if '.' in name: raise KeyError("module name can't contain \".\"")
        if name == '': raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def _get_named_parameters(self, prefix=''):
        """Helper to recursively get named parameters."""
        memo = set()
        for name, param in self._parameters.items():
            if param is not None and param not in memo:
                memo.add(param)
                yield prefix + ('.' if prefix else '') + name, param

        for name, module in self._modules.items():
            if module is not None:
                yield from module._get_named_parameters(prefix + ('.' if prefix else '') + name)

    def parameters(self):
        """
        Return an iterable (generator) of all Value parameters
        in this module and its submodules.
        """
        # yield only the Value objects (ignore names)
        for name, param in self._get_named_parameters():
            yield param

    def zero_grad(self):
        """Sets gradients (.grad attribute) of all parameters to zero."""
        for p in self.parameters():
            if p is None:
                continue
            # Ensure grad exists and set to zeros of correct shape
            p.grad = np.zeros_like(p.data, dtype=np.float64)

    def __call__(self, *args, **kwargs):
        """Defines the forward pass. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward pass (__call__)")

    def state_dict(self):
        """Returns a dictionary containing the module's state (parameter data)."""
        return {name: param.data for name, param in self._get_named_parameters()}

    def save_state_dict(self, filepath):
        current_state_dict = self.state_dict()
        try:
            np.savez_compressed(filepath, **current_state_dict)
            print(f"State dictionary saved to {filepath}")
        except Exception as e:
            print(f"Error saving state dictionary: {e}")

    def load_state_dict(self, filepath):
        try:
            loaded_state_dict_npz = np.load(filepath)
            loaded_state_dict = {k: loaded_state_dict_npz[k] for k in loaded_state_dict_npz.files}
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return
        except Exception as e:
            print(f"Error loading state dictionary: {e}")
            return

        current_params_dict = dict(self._get_named_parameters())
        print(f"Loading state dictionary from {filepath}...")
        loaded_keys = set(loaded_state_dict.keys())
        model_keys = set(current_params_dict.keys())

        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys
        if missing_keys: print(f"Warning: Missing keys in state_dict: {missing_keys}")
        if unexpected_keys: print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

        for name, param in current_params_dict.items():
            if name in loaded_state_dict:
                loaded_array = loaded_state_dict[name]
                if param.data.shape != loaded_array.shape:
                    print(f"Error: Shape mismatch for parameter '{name}'. "
                          f"Model has {param.data.shape}, loaded has {loaded_array.shape}.")
                    continue
                param.data[:] = loaded_array

        print("State dictionary loaded successfully (check warnings/errors above).")

    def __setattr__(self, name, value):
         """Override setattr to automatically register Modules and Parameters."""
         if name.startswith('_') or not (isinstance(value, Value) or isinstance(value, Module)):
             super().__setattr__(name, value)
             return

         if isinstance(value, Value):
             if '_parameters' not in self.__dict__:
                 raise AttributeError("cannot assign parameter before Module.__init__() call")
             self._modules.pop(name, None)
             self.register_parameter(name, value)

         elif isinstance(value, Module):
             if '_modules' not in self.__dict__:
                 raise AttributeError("cannot assign module before Module.__init__() call")
             self._parameters.pop(name, None)
             self.add_module(name, value)

         super().__setattr__(name, value)
