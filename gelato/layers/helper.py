import pymc3 as pm
from lasagne.layers.helper import *


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
