from __future__ import print_function
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


class TestSpecs(unittest.TestCase):
    # they require arguments
    # TODO: add arguments to spec init
    dont_test = [
        'BetaSpec',
        'ExponentialSpec',
        'LaplaceSpec',
        'StudentTSpec',
        'CauchySpec',
        'HalfCauchySpec',
        'GammaSpec',
        'WeibullSpec',
        'HalfStudentTSpec',
        'ChiSquaredSpec',
        'WaldSpec',
        'ParetoSpec',
        'InverseGammaSpec',
        'ExGaussianSpec',
        'VonMisesSpec',
        'NormalMixtureSpec'
    ]

    def setUp(self):
        from ..spec import __all__, PartialSpec
        from gelato import spec
        self.specs = [
            getattr(spec, s)
            for s in __all__
            if (s.endswith('Spec') and
                issubclass(getattr(spec, s), PartialSpec) and
                not getattr(spec, s) is PartialSpec and
                s not in self.dont_test)
        ]

    def test_all(self):
        for spec in self.specs:
            print('testing {!r}'.format(spec))
            with Model():
                self.assertEqual(
                    spec()((1, 1)).tag.test_value.shape,
                    (1, 1)
                )
                self.assertEqual(
                    spec()((10, 1)).tag.test_value.shape,
                    (10, 1)
                )
            print('{!r} -- ok'.format(spec))
if __name__ == '__main__':
    unittest.main()
