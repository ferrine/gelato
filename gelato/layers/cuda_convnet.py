import sys
import lasagne.layers.cuda_convnet as __cloned
from .base import bayes
__module = sys.modules[__name__]
del sys
__all__ = [
    "Conv2DCCLayer",
    "MaxPool2DCCLayer",
    "ShuffleBC01ToC01BLayer",
    "bc01_to_c01b",
    "ShuffleC01BToBC01Layer",
    "c01b_to_bc01",
    "NINLayer_c01b",
]
for obj_name in __all__:
    try:
        setattr(__module, obj_name, bayes(getattr(__cloned, obj_name)))
    except TypeError:
        setattr(__module, obj_name, getattr(__cloned, obj_name))
