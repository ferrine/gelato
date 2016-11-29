import six
from lasagne.layers.cuda_convnet import *
from .base import LayerModelMeta

__all__ = [
    'BayesianNINLayer_c01b',
    'BayesianConv2DCCLayer'
]


class BayesianNINLayer_c01b(
        six.with_metaclass(LayerModelMeta, NINLayer_c01b)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=NINLayer_c01b.__name__,
        doc=NINLayer_c01b.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianConv2DCCLayer(
        six.with_metaclass(LayerModelMeta, Conv2DCCLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=Conv2DCCLayer.__name__,
        doc=Conv2DCCLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError
