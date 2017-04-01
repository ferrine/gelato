import pytest
from pymc3.model import Model
from gelato.layers.base import Layer, MergeLayer


class BLayer(Layer):
    def __init__(self, incomming, name=None):
        super(BLayer, self).__init__(incomming, name)
        self.W = self.add_param(None, (10, 10), name='W')


class BMLayer(MergeLayer):
    def __init__(self, incommings, name=None):
        super(BMLayer, self).__init__(incommings, name)
        self.W = self.add_param(None, (10, 10))


class TestBasicLayers(object):
    def test_simple_init_layer(self):
        with Model():
            l = BLayer((10, 10))
            assert isinstance(l.name, str)
            l = BLayer((10, 10), 'l2')
            assert (len(l.vars) == 1)
            assert isinstance(l.name, str)
            l = BMLayer([(10, 10)], name='merge')
            assert isinstance(l.name, str)
            assert (len(l.vars) == 1)
            l = BMLayer([(10, 10)])
            assert isinstance(l.name, str)

        with pytest.raises(TypeError):
            BMLayer((10, 10))
            BLayer((10, 10))

