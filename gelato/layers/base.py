from pymc3 import Model, Normal
from lasagne.layers.base import Layer, MergeLayer
from gelato.spec import DistSpec


class BayesianLayer(Model, Layer):
    """Base layer as proposed in [1]_

    References
    ----------
    .. [1] Charles Blundell et al: "Weight Uncertainty in Neural Networks"
        arXiv preprint arXiv:1505.05424
    """
    default_spec = DistSpec(Normal, mu=0, sd=0.001)

    def __init__(self, incomming, name=None, model=None):
        if name is None:
            name = '{}_{}'.format(self.__class__.__name__, id(self))
        Model.__init__(self, name, model)
        Layer.__init__(self, incomming, name)

    def __new__(cls, *args, **kwargs):
        instance = super(BayesianLayer, cls).__new__(cls, *args, **kwargs)
        if instance.isroot:
            raise TypeError('Unable to init as root model')
        return instance

    def add_param(self, spec, shape, name=None, **tags):
        if not isinstance(spec, DistSpec):
            spec = self.default_spec
        if name is not None:
            spec = spec.with_name(name)
        return super(BayesianLayer, self).add_param(spec, shape, **tags)


class BayesianMergeLayer(BayesianLayer, MergeLayer):
    """Base Merge layer as proposed in [1]_

    References
    ----------
    .. [1] Charles Blundell et al: "Weight Uncertainty in Neural Networks"
        arXiv preprint arXiv:1505.05424
    """
    def __init__(self, incommings, name=None, model=None):
        if name is None:
            name = '{}_{}'.format(self.__class__.__name__, id(self))
        Model.__init__(self, name, model)
        MergeLayer.__init__(self, incommings, name)
