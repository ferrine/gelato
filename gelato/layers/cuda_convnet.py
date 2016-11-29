import six
from lasagne.layers.cuda_convnet import *
from .base import LayerModelMeta

__all__ = [
    'BayesianNINLayer_c01b',
    'BayesianConv2DCCLayer'
]


class BayesianNINLayer_c01b(
        six.with_metaclass(LayerModelMeta, NINLayer_c01b)):
    pass


class BayesianConv2DCCLayer(
        six.with_metaclass(LayerModelMeta, Conv2DCCLayer)):
    pass
