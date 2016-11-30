import unittest
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

if __name__ == '__main__':
    unittest.main()
