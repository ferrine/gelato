import sys
import lasagne.layers.normalization as __cloned
from lasagne import nonlinearities
from .base import bayes as __bayes
__module = sys.modules[__name__]
del sys
__all__ = __cloned.__all__
for obj_name in __all__:
    try:
        setattr(__module, obj_name, __bayes(getattr(__cloned, obj_name)))
    except TypeError:
        setattr(__module, obj_name, getattr(__cloned, obj_name))


# override lasagne batch_norm
# to use galato layers
def batch_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn_name = (kwargs.pop('name', None) or
               (getattr(layer, 'name', None) and layer.name + '_bn'))
    layer = BatchNormLayer(layer, name=bn_name, **kwargs)
    if nonlinearity is not None:
        from .special import NonlinearityLayer
        nonlin_name = bn_name and bn_name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
    return layer
batch_norm.__doc__ = __cloned.batch_norm.__doc__
