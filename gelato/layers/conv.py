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
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=Conv1DLayer.__name__,
        doc=Conv1DLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianConv2DLayer(
        six.with_metaclass(LayerModelMeta, Conv2DLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=Conv2DLayer.__name__,
        doc=Conv2DLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianTransposedConv2DLayer(
        six.with_metaclass(LayerModelMeta, TransposedConv2DLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=TransposedConv2DLayer.__name__,
        doc=TransposedConv2DLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianDeconv2DLayer(
        six.with_metaclass(LayerModelMeta, Deconv2DLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=Deconv2DLayer.__name__,
        doc=Deconv2DLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianDilatedConv2DLayer(
        six.with_metaclass(LayerModelMeta, DilatedConv2DLayer)):
    __doc__ = """Bayesian{clsname}\n\n{doc}""".format(
        clsname=DilatedConv2DLayer.__name__,
        doc=DilatedConv2DLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError
