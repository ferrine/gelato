import unittest
from pymc3 import Normal, Lognormal, Model
from gelato.spec import DistSpec


class TestSpec(unittest.TestCase):
    def test_shape(self):
        spec = DistSpec(Normal, mu=0, sd=1)
        spec2 = DistSpec(Normal, mu=0, sd=DistSpec(Lognormal, 0, 1))

        with Model('layer'):
            var = spec((100, 100), 'var')
            var2 = spec2((100, 100), 'var2')
            self.assertEqual(var.init_value.shape, (100, 100))
            self.assertTrue(var.name.endswith('var'))
            self.assertEqual(var2.init_value.shape, (100, 100))
            self.assertTrue(var2.name.endswith('var2'))

if __name__ == '__main__':
    unittest.main()
