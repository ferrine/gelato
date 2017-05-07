import pymc3 as pm
from lasagne.layers.helper import *
from lasagne.layers.helper import __all__ as __helper__all__
__all__ = [
    "find_parent",
    "find_root",
] + __helper__all__


def find_parent(layer):
    candidates = get_all_layers(layer)[::-1]
    found = None
    for candidate in candidates:
        if isinstance(candidate, pm.Model):
            found = candidate
            break
    return found


def find_root(layer):
    model = find_parent(layer)
    if model is not None:
        return model.root
    else:
        return None
