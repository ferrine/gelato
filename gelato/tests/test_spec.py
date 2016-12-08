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
    for_test = dict(
        UniformSpec=[
            dict(),
            dict(lower=0, upper=1)
        ],
        NormalSpec=[
            dict(),
            dict(mu=0, sd=1),
        ],
        BetaSpec=[
            dict(),
            dict(alpha=1, beta=2)
        ],
        ExponentialSpec=[
            dict(),
            dict(lam=10)
        ],
        LaplaceSpec=[
            dict(),
            dict(mu=2, b=3)
        ],
        StudentTSpec=[
            dict(nu=10),
            dict(nu=10, mu=5, sd=10),
        ],
        CauchySpec=[
            dict(),
            dict(alpha=2, beta=5)
        ],
        HalfCauchySpec=[
            dict(beta=5)
        ],
        GammaSpec=[
            dict(alpha=2, beta=5),
        ],
        WeibullSpec=[
            dict(alpha=2, beta=5)
        ],
        HalfStudentTSpec=[
            dict(nu=10),
            dict(nu=10, mu=5, sd=10),
        ],
        LognormalSpec=[
            dict(),
            dict(mu=0, sd=1),
        ],
        ChiSquaredSpec=[
            dict(nu=10)
        ],
        HalfNormalSpec=[
            dict(),
            dict(sd=1),
        ],
        WaldSpec=[
            dict(mu=1, lam=1, alpha=2),
        ],
        ParetoSpec=[
            dict(alpha=.5, m=2)
        ],
        InverseGammaSpec=[
            dict(alpha=2),
            dict(alpha=.5, beta=2)
        ],
        ExGaussianSpec=[
            dict(mu=0, sd=1, nu=2),
        ],
        VonMisesSpec=[
            dict(mu=2, kappa=1)
        ],
        SkewNormalSpec=[
            dict(),
            dict(mu=0, sd=1, alpha=2),
        ],
        NormalMixtureSpec=[
            dict(w=[.1, .9], mu=[0, 0], sd=[.1, 10]),
        ]
    )
    skip = [
        'HalfStudentTSpec',
        'NormalMixtureSpec'
    ]

    def setUp(self):
        from ..spec import __all__
        from gelato import spec
        self.specs = [
            getattr(spec, s)
            for s in __all__
            if s in self.for_test and s not in self.skip
        ]

    def test_all(self):
        for spec in self.specs:
            try:
                with Model():
                    for kwargs in self.for_test[spec.__name__]:
                        self.assertEqual(
                            spec(**kwargs)((1, 1)).tag.test_value.shape,
                            (1, 1)
                        )
                        self.assertEqual(
                            spec(**kwargs)((10, 1)).tag.test_value.shape,
                            (10, 1)
                        )
                        self.assertEqual(
                            spec(**kwargs)((10, 1, 10)).tag.test_value.shape,
                            (10, 1, 10)
                        )
            except Exception as e:
                print('{} -- fail'.format(spec.__name__))
                raise e
            else:
                print('.', end='')

if __name__ == '__main__':
    unittest.main()
