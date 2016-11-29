import six
from lasagne.layers.local import *
from .base import LayerModelMeta

__all__ = [
    "BayesianLocallyConnected2DLayer",
]


class BayesianLocallyConnected2DLayer(
        six.with_metaclass(LayerModelMeta, LocallyConnected2DLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=LocallyConnected2DLayer.__name__,
        doc=LocallyConnected2DLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError
