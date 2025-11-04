# my_ml_lib/nn/modules/containers.py
from .base import Module
from collections import OrderedDict # Use OrderedDict if allowing dict input
from ..autograd import Value # Import Value for type hinting

class Sequential(Module):
    """
    A sequential container for stacking Modules.
    Modules will be executed in the order they are added.
    """
    def __init__(self, *args):
        """
        Initializes the Sequential container.

        Args:
            *args (Module): Sequence of modules to add.
                            Can also be an OrderedDict.
        """
        #  Initialize base class and register modules ---
        super().__init__() # CRITICAL: Initializes _parameters and _modules
        if len(args) == 1 and isinstance(args[0], OrderedDict):
             # If an OrderedDict is passed, add modules using keys as names
             for key, module in args[0].items():
                 # Check if the value is actually a Module
                 if not isinstance(module, Module):
                     raise TypeError(f"Value for key '{key}' is not an nn.Module subclass: {type(module)}")
                 self.add_module(key, module) # Registers in self._modules
        else:
             # If positional arguments are passed, add modules using indices as names
             for i, module in enumerate(args):
                 # Check if the argument is actually a Module
                 if not isinstance(module, Module):
                     raise TypeError(f"Argument {i} is not an nn.Module subclass: {type(module)}")
                 self.add_module(str(i), module) # Registers in self._modules

        # --- TODO: Understand the registration process ---
        # Notice how `add_module` (defined in the base Module class) is used here.
        # This populates the `self._modules` OrderedDict, which is essential for
        # parameter discovery (`parameters()`, `state_dict()`) and the forward pass.


    def __call__(self, x: Value) -> Value:
        """
        Defines the forward pass through the sequential layers.

        Args:
            x (Value): Input Value object.

        Returns:
            Value: Output of the final layer in the sequence.
        """
        # --- TODO: Implement the sequential forward pass ---
        # Iterate through the modules stored in this container.
        # Pass the output of one module as the input to the next.
        #
        # Hint: The registered modules are stored in the `self._modules` OrderedDict.
        #       You can iterate through its values: `for module in self._modules.values():`
        #       Keep track of the current output `x` and update it in each step.
        
        for module in self._modules.values():
            x = module(x)
        return x

   

        

    # --- String Representation ---
    def __repr__(self):
        """Provides a developer-friendly string representation of the container."""
        # This iterates through the registered modules and their names
        layer_strs = [f"  ({name}): {module}" for name, module in self._modules.items()]
        return f"Sequential(\n" + "\n".join(layer_strs) + "\n)"
