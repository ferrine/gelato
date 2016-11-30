import pymc3 as pm
import pymc3.distributions.distribution
import functools

__all__ = [
    'get_default_spec',
    'set_default_spec',
    'DistSpec',
    'PartialSpec',
    'UniformSpec',
    'NormalSpec',
    'BetaSpec',
    'ExponentialSpec',
    'LaplaceSpec',
    'StudentTSpec',
    'CauchySpec',
    'HalfCauchySpec',
    'GammaSpec',
    'WeibullSpec',
    'HalfStudentTSpec',
    'LognormalSpec',
    'ChiSquaredSpec',
    'HalfNormalSpec',
    'WaldSpec',
    'ParetoSpec',
    'InverseGammaSpec',
    'ExGaussianSpec',
    'VonMisesSpec',
    'SkewNormalSpec',
    'NormalMixtureSpec'
]


class DistSpec(object):
    """Spec based on pymc3 distributions

    Parameters
    ----------
    distcls : pymc3.Distribution
    args : args for `distcls`
    kwargs : kwargs for `distcls`

    Usage
    -----
    spec = DistSpec(Normal, mu=0, sd=DistSpec(Lognormal, 0, 1))
    """

    def __init__(self, distcls, *args, **kwargs):
        if not issubclass(
                distcls,
                pymc3.distributions.distribution.Distribution):
            raise ValueError('We can deal with pymc3 '
                             'distributions only, got {!r} instead'
                             .format(distcls))
        self.args = args
        self.kwargs = kwargs
        self.distcls = distcls

    def __call__(self, shape, name=None):
        model = pm.modelcontext(None)
        if name is None:
            name = 'w{}'.format(len(model.vars))
        called_args = self._call_args(self.args, name, shape)
        called_kwargs = self._call_kwargs(self.kwargs, name, shape)
        called_kwargs.update(shape=shape)
        val = model.Var(
                name, self.distcls.dist(
                    *called_args,
                    **called_kwargs
                ),
            )
        val.tag.test_value = val.random().reshape(shape)
        return val

    def with_name(self, name):
        return functools.partial(self, name=name)

    def _call_args(self, args, name, shape):
        return [
            self._call(arg, '{}_arg{}'.format(name, i), shape)
            for i, arg in enumerate(args)
        ]

    def _call_kwargs(self, kwargs, name, shape):
        return {
            key: self._call(arg, '{}_{}'.format(name, key), shape)
            for key, arg in kwargs.items()
        }

    @staticmethod
    def _call(arg, label, shape):
        if callable(arg):
            if isinstance(arg, DistSpec):
                return arg(shape, label)
            else:
                raise TypeError(
                    'Cannot proceed type {} in DistSpec'
                    .format(type(arg))
                )
        else:
            return arg

    def __repr__(self):
        template = '<{cls}: {args!r}; {kwargs!r}>'
        return template.format(cls=self.distcls.__name__,
                               args=self.args,
                               kwargs=self.kwargs)


_default_spec = DistSpec(pm.Normal, mu=0, sd=10)


def get_default_spec():
    return _default_spec


def set_default_spec(spec):
    global _default_spec
    _default_spec = spec


class PartialSpec(DistSpec):
    spec = None

    def __init__(self, *args, **kwargs):
        super(PartialSpec, self).__init__(self.spec, *args, **kwargs)


class UniformSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Uniform.__name__,
        doc=pm.Uniform.__doc__
    )
    spec = pm.Uniform


class NormalSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Normal.__name__,
        doc=pm.Normal.__doc__
    )
    spec = pm.Normal


class BetaSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Beta.__name__,
        doc=pm.Beta.__doc__
    )
    spec = pm.Beta


class ExponentialSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Exponential.__name__,
        doc=pm.Exponential.__doc__
    )
    spec = pm.Exponential


class LaplaceSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Laplace.__name__,
        doc=pm.Laplace.__doc__
    )
    spec = pm.Laplace


class StudentTSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.StudentT.__name__,
        doc=pm.StudentT.__doc__
    )
    spec = pm.StudentT


class CauchySpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Cauchy.__name__,
        doc=pm.Cauchy.__doc__
    )
    spec = pm.Cauchy


class HalfCauchySpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.HalfCauchy.__name__,
        doc=pm.HalfCauchy.__doc__
    )
    spec = pm.HalfCauchy


class GammaSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Gamma.__name__,
        doc=pm.Gamma.__doc__
    )
    spec = pm.Gamma


class WeibullSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Weibull.__name__,
        doc=pm.Weibull.__doc__
    )
    spec = pm.Weibull


class HalfStudentTSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.HalfStudentT.distribution.__name__,
        doc="""Bounded StudentT with support on [0, +inf]\n{doc}""".format(
            doc=pm.StudentT.__doc__
        )
    )
    spec = pm.HalfStudentT


class LognormalSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Lognormal.__name__,
        doc=pm.Lognormal.__doc__
    )
    spec = pm.Lognormal


class ChiSquaredSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.ChiSquared.__name__,
        doc=pm.ChiSquared.__doc__
    )
    spec = pm.ChiSquared


class HalfNormalSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.HalfNormal.__name__,
        doc=pm.HalfNormal.__doc__
    )
    spec = pm.HalfNormal


class WaldSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Wald.__name__,
        doc=pm.Wald.__doc__
    )
    spec = pm.Wald


class ParetoSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.Pareto.__name__,
        doc=pm.Pareto.__doc__
    )
    spec = pm.Pareto


class InverseGammaSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.InverseGamma.__name__,
        doc=pm.InverseGamma.__doc__
    )
    spec = pm.InverseGamma


class ExGaussianSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.ExGaussian.__name__,
        doc=pm.ExGaussian.__doc__
    )
    spec = pm.ExGaussian


class VonMisesSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.VonMises.__name__,
        doc=pm.VonMises.__doc__
    )
    spec = pm.VonMises


class SkewNormalSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.SkewNormal.__name__,
        doc=pm.SkewNormal.__doc__
    )
    spec = pm.SkewNormal


class NormalMixtureSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.NormalMixture.__name__,
        doc=pm.NormalMixture.__doc__
    )
    spec = pm.NormalMixture
