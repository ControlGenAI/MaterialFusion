import functools
from inspect import signature
from torch.utils import checkpoint

use_mode: bool = True


def use_grad_checkpointing(mode: bool=True):
    global use_mode
    use_mode = mode


def checkpoint_forward(func):
    sig = signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if use_mode:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            new_args = bound.arguments.values()
            result = checkpoint.checkpoint(func, *new_args, use_reentrant=False)
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper
