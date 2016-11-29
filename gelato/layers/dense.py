import six
from lasagne.layers.dense import *
from .base import LayerModelMeta

__all__ = [
    "BayesianDenseLayer",
    "BayesianNINLayer",
]


class BayesianDenseLayer(
        six.with_metaclass(LayerModelMeta, DenseLayer)):
    pass


class BayesianNINLayer(
        six.with_metaclass(LayerModelMeta, NINLayer)):
    pass
