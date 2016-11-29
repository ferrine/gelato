import six
from lasagne.layers.corrmm import *
from .base import LayerModelMeta

__all__ = [
    'BayesianConv2DMMLayer'
]


class BayesianConv2DMMLayer(
        six.with_metaclass(LayerModelMeta, Conv2DMMLayer)):
    pass
