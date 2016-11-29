import six
from lasagne.layers.corrmm import *
from .base import LayerModelMeta

__all__ = [
    'BayesianConv2DMMLayer'
]


class BayesianConv2DMMLayer(
        six.with_metaclass(LayerModelMeta, Conv2DMMLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=Conv2DMMLayer.__name__,
        doc=Conv2DMMLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError
