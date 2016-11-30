import sys
import lasagne.layers.pool as __cloned
from .base import bayes as __bayes
__module = sys.modules[__name__]
del sys
__all__ = [
    "MaxPool1DLayer",
    "MaxPool2DLayer",
    "Pool1DLayer",
    "Pool2DLayer",
    "Upscale1DLayer",
    "Upscale2DLayer",
    "Upscale3DLayer",
    "FeaturePoolLayer",
    "FeatureWTALayer",
    "GlobalPoolLayer",
]
for obj_name in __all__:
    try:
        setattr(__module, obj_name, __bayes(getattr(__cloned, obj_name)))
    except TypeError:
        setattr(__module, obj_name, getattr(__cloned, obj_name))
