import unittest
from pymc3.model import Model
from lasagne.layers import DenseLayer, LSTMLayer

from gelato.layers.base import BayesianLayer, BayesianMergeLayer
from gelato.layers.utils import islayersub, ismergesub


class BLayer(BayesianLayer):
    def __init__(self, incomming, name=None, model=None):
        super(BLayer, self).__init__(incomming, name, model)
        self.W = self.add_param(None, (10, 10), name='W')


class BMLayer(BayesianMergeLayer):
    def __init__(self, incommings, name=None, model=None):
        super(BMLayer, self).__init__(incommings, name, model)
        self.W = self.add_param(None, (10, 10))


class TestBasicLayers(unittest.TestCase):
    def test_simple_init_layer(self):
        with Model():
            l = BLayer((10, 10))
            self.assertIsInstance(l.name, str)
            l = BLayer((10, 10), 'l2')
            self.assertEqual(len(l.vars), 1)
            self.assertIsInstance(l.name, str)
            l = BMLayer([(10, 10)], name='merge')
            self.assertIsInstance(l.name, str)
            self.assertEqual(len(l.vars), 1)
            l = BMLayer([(10, 10)])
            self.assertIsInstance(l.name, str)

        self.assertRaises(TypeError, BMLayer, (10, 10))
        self.assertRaises(TypeError, BLayer, (10, 10))

    def test_issubclass(self):
        self.assertTrue(islayersub(BLayer))
        self.assertTrue(islayersub(DenseLayer))

        self.assertFalse(islayersub(LSTMLayer))
        self.assertFalse(islayersub(BMLayer))

        self.assertTrue(ismergesub(BMLayer))
        self.assertTrue(ismergesub(LSTMLayer))

        self.assertFalse(ismergesub(BLayer))
        self.assertFalse(ismergesub(DenseLayer))

if __name__ == '__main__':
    unittest.main()
