import six
from lasagne.layers.dense import *
from .base import LayerModelMeta

__all__ = [
    "BayesianDenseLayer",
    "BayesianNINLayer",
]


class BayesianDenseLayer(
        six.with_metaclass(LayerModelMeta, DenseLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=DenseLayer.__name__,
        doc=DenseLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianNINLayer(
        six.with_metaclass(LayerModelMeta, NINLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=NINLayer.__name__,
        doc=NINLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError
