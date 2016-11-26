import unittest
from pymc3.model import Model
from gelato.layers.base import BayesianLayer, BayesianMergeLayer


class BLayer(BayesianLayer):
    def __init__(self, incomming, name=None, model=None):
        super(BLayer, self).__init__(incomming, name, model)
        self.W = self.add_param(None, (10, 10))


class BMLayer(BayesianMergeLayer):
    def __init__(self, incommings, name=None, model=None):
        super(BMLayer, self).__init__(incommings, name, model)
        self.W = self.add_param(None, (10, 10))


class TestBasicLayers(unittest.TestCase):
    def test_simple_init_layer(self):
        with Model():
            layer = BLayer((10, 10))
            mlayer = BMLayer([(10, 10)], name='merge')
