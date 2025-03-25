# joeyNMT/common.py

import torch.nn as nn

class DataParallelWrapper(nn.Module):
    """
    DataParallel wrapper to pass through the model attributes

    Example:
    1) For DataParallel:
        >>> from torch.nn import DataParallel as DP
        >>> model = DataParallelWrapper(DP(model))

    2) For DistributedDataParallel:
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> model = DataParallelWrapper(DDP(model))
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        assert hasattr(module, "module"), "Wrapped module must have 'module' attribute"
        self.module = module

    def __getattr__(self, name):
        """Forward missing attributes to twice-wrapped module."""
        try:
            # Defer to nn.Module's logic
            return super().__getattr__(name)
        except AttributeError:
            try:
                # Forward to the once-wrapped module (DP/DDP)
                return getattr(self.module, name)
            except AttributeError:
                # Forward to the original module inside DP/DDP
                return getattr(self.module.module, name)

    def state_dict(self, *args, **kwargs):
        """Save the state_dict of the original module."""
        return self.module.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Load the state_dict into the original module."""
        self.module.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)
