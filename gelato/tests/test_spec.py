import unittest
from pymc3 import Normal, Model
from gelato.spec import DistSpec


class TestSpec(unittest.TestCase):
    def test_shape(self):
        spec = DistSpec(Normal, mu=0, sd=1)
        with Model('layer'):
            var = spec((100, 100), 'var')
            self.assertEqual(var.init_value.shape, (100, 100))
            self.assertTrue(var.name.endswith('var'))

if __name__ == '__main__':
    unittest.main()
