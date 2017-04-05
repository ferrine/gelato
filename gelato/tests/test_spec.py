import pytest
import numpy as np
from pymc3 import Normal, Lognormal, Model

from gelato.specs.base import DistSpec
from gelato.specs import *


class TestSpec(object):
    def test_shape(self):
        spec = DistSpec(Normal, mu=0, sd=1)
        spec2 = DistSpec(Normal, mu=0, sd=DistSpec(Lognormal, 0, 1))

        with Model('layer'):
            var = spec((100, 100), 'var')
            var2 = spec2((100, 100), 'var2')
            assert (var.init_value.shape == (100, 100))
            assert (var.name.endswith('var'))
            assert (var2.init_value.shape == (100, 100))
            assert (var2.name.endswith('var2'))


_for_test = dict(
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
_skip = [
       'HalfStudentTSpec',
       'NormalMixtureSpec'
]


def setup_specs_kwargs():
    from gelato.specs.dist import __all__
    from gelato.specs import dist
    specs = [
        getattr(dist, s)
        for s in __all__
        if s in set(_for_test.keys()) - set(_skip)
    ]
    specs_kwargs = []
    for spec in specs:
        specs_kwargs.extend([(spec, d) for d in _for_test[spec.__name__]])
    return specs_kwargs


@pytest.mark.parametrize(
    ['spec', 'kwargs'],
    setup_specs_kwargs()
)
def test_spec(spec, kwargs):
    with Model():
        assert (
            spec(**kwargs)((1, 1)).tag.test_value.shape
            ==
            (1, 1)
        )
        assert (
            spec(**kwargs)((10, 1)).tag.test_value.shape
            ==
            (10, 1)
        )
        assert (
            spec(**kwargs)((10, 1, 10)).tag.test_value.shape
            ==
            (10, 1, 10)
        )


def pseudo_random(shape):
    np.random.seed(sum(shape))
    return np.random.randint(1, 5, size=shape).astype('int32')

binary = [
    '__add__',
    '__sub__',
    '__mul__',
    '__truediv__',
    '__floordiv__',
    '__pow__',
    '__and__',
    '__or__',
    '__xor__',
    '__lt__',
    '__gt__',
    '__ge__',
    '__le__',
]

unary = [
   '__neg__',
]


@pytest.mark.parametrize(
    'op',
    unary
)
def test_unary_op(op):
    test_op = getattr(np.ndarray, op)
    eval_op = getattr(DistSpec, op)
    shape = (3, 3)
    a_test = pseudo_random(shape)
    c_test = test_op(a_test)
    with Model():
        a = DistSpec(Normal, testval=pseudo_random).astype(dtype='int32')
        c = eval_op(a)
        c_val = c(shape).tag.test_value
    np.testing.assert_allclose(c_test, c_val)


@pytest.mark.parametrize(
    'op',
    binary
)
def test_binary_op(op):
    test_op = getattr(np.ndarray, op)
    eval_op = getattr(DistSpec, op)
    shape = (3, 3)
    a_test = pseudo_random(shape)
    b_test = pseudo_random(shape)
    c_test = test_op(a_test, b_test)
    with Model():
        a = DistSpec(Normal, testval=pseudo_random).astype(dtype='int32')
        b = DistSpec(Normal, testval=pseudo_random).astype(dtype='int32')
        c = eval_op(a, b)
        c_val = c(shape).tag.test_value
    np.testing.assert_allclose(c_test, c_val)


def test_expressions():
    with Model() as model:
        expr = (NormalSpec() + LaplaceSpec()) / 100 - NormalSpec()
        var = expr((10, 10))
        assert var.tag.test_value.shape == (10, 10)
        assert len(model.free_RVs) == 3
