import six
from lasagne.layers.local import *
from .base import LayerModelMeta

__all__ = [
    "BayesianLocallyConnected2DLayer",
]


class BayesianLocallyConnected2DLayer(
        six.with_metaclass(LayerModelMeta, LocallyConnected2DLayer)):
    pass
