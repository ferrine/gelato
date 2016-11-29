import six
from lasagne.layers.dnn import *
from .base import LayerModelMeta

__all__ = [
    'BayesianConv2DDNNLayer',
    'BayesianConv3DDNNLayer'
]


class BayesianConv2DDNNLayer(
        six.with_metaclass(LayerModelMeta, Conv2DDNNLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=Conv2DDNNLayer.__name__,
        doc=Conv2DDNNLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianConv3DDNNLayer(
        six.with_metaclass(LayerModelMeta, Conv3DDNNLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=Conv3DDNNLayer.__name__,
        doc=Conv3DDNNLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError
