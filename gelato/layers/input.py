import sys
from lasagne.layers.input import __all__
import lasagne.layers.input as __cloned
from .base import bayes as __bayes
__module = sys.modules[__name__]
del sys
for obj_name in __all__:
    try:
        setattr(__module, obj_name, __bayes(getattr(__cloned, obj_name)))
    except TypeError:
        setattr(__module, obj_name, getattr(__cloned, obj_name))
