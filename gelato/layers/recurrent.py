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
    pass


class BayesianRecurrentLayer(
        six.with_metaclass(LayerModelMeta, RecurrentLayer)):
    pass


class BayesianGate(
        six.with_metaclass(LayerModelMeta, Gate)):
    pass


class BayesianLSTMLayer(
        six.with_metaclass(LayerModelMeta, LSTMLayer)):
    pass


class BayesianGRULayer(
        six.with_metaclass(LayerModelMeta, GRULayer)):
    pass
