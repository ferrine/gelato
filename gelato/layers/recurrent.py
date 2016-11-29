import six
from lasagne.layers.recurrent import *
from .base import LayerModelMeta

__all__ = [
    "BayesianCustomRecurrentLayer",
    "BayesianRecurrentLayer",
    "BayesianGate",
    "BayesianLSTMLayer",
    "BayesianGRULayer"
]


class BayesianCustomRecurrentLayer(
        six.with_metaclass(LayerModelMeta, CustomRecurrentLayer)):
    __doc__ = """Gelato Bayesian{clsname}\n\n{doc}""".format(
        clsname=CustomRecurrentLayer.__name__,
        doc=CustomRecurrentLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianRecurrentLayer(
        six.with_metaclass(LayerModelMeta, RecurrentLayer)):
    __doc__ = """Gelato Bayesian{clsname}\n\n{doc}""".format(
        clsname=RecurrentLayer.__name__,
        doc=RecurrentLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianGate(
        six.with_metaclass(LayerModelMeta, Gate)):
    __doc__ = """Gelato Bayesian{clsname}\n\n{doc}""".format(
        clsname=Gate.__name__,
        doc=Gate.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianLSTMLayer(
        six.with_metaclass(LayerModelMeta, LSTMLayer)):
    __doc__ = """Gelato Bayesian{clsname}\n\n{doc}""".format(
        clsname=LSTMLayer.__name__,
        doc=LSTMLayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError


class BayesianGRULayer(
        six.with_metaclass(LayerModelMeta, GRULayer)):
    __doc__ = """Gelato Bayesian{clsname}\n\n{doc}""".format(
        clsname=GRULayer.__name__,
        doc=GRULayer.__doc__
    )

    def __getattr__(self, item):
        raise AttributeError
