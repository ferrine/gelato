import six
from lasagne.layers.conv import *
from .base import LayerModelMeta

__all__ = [
    "BayesianConv1DLayer",
    "BayesianConv2DLayer",
    "BayesianTransposedConv2DLayer",
    "BayesianDeconv2DLayer",
    "BayesianDilatedConv2DLayer",
]


class BayesianConv1DLayer(
        six.with_metaclass(LayerModelMeta, Conv1DLayer)):
    pass


class BayesianConv2DLayer(
        six.with_metaclass(LayerModelMeta, Conv2DLayer)):
    pass


class BayesianTransposedConv2DLayer(
        six.with_metaclass(LayerModelMeta, TransposedConv2DLayer)):
    pass


class BayesianDeconv2DLayer(
        six.with_metaclass(LayerModelMeta, Deconv2DLayer)):
    pass


class BayesianDilatedConv2DLayer(
        six.with_metaclass(LayerModelMeta, DilatedConv2DLayer)):
    pass
