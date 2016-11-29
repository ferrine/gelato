import six
from lasagne.layers.dnn import *
from .base import LayerModelMeta

__all__ = [
    'BayesianConv2DDNNLayer',
    'BayesianConv3DDNNLayer'
]


class BayesianConv2DDNNLayer(
        six.with_metaclass(LayerModelMeta, Conv2DDNNLayer)):
    pass


class BayesianConv3DDNNLayer(
        six.with_metaclass(LayerModelMeta, Conv3DDNNLayer)):
    pass
