import sys
import lasagne.layers.dense as __cloned
from .base import bayes as __bayes
__module = sys.modules[__name__]
del sys
__all__ = []
for obj_name in __cloned.__all__:
    try:
        setattr(__module, obj_name, __bayes(getattr(__cloned, obj_name)))
        __all__ += [obj_name]
    except TypeError:
        setattr(__module, obj_name, getattr(__cloned, obj_name))
