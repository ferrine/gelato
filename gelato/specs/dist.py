import copy

import pymc3 as pm

from gelato.specs.base import DistSpec, get_default_testval

__all__ = [
    'get_default_spec',
    'set_default_spec',
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
    'LognormalSpec',
    'ChiSquaredSpec',
    'HalfNormalSpec',
    'WaldSpec',
    'ParetoSpec',
    'InverseGammaSpec',
    'ExGaussianSpec',
    'VonMisesSpec',
    'SkewNormalSpec',
    # 'HalfStudentTSpec',
    # 'NormalMixtureSpec'
]

_default_spec = DistSpec(pm.Normal, mu=0, sd=10)


def get_default_spec(testval=None):
    # to avoid init collision
    cp = copy.deepcopy(_default_spec)
    if testval is None:
        cp.testval = get_default_testval()
    else:
        cp.testval = testval
    return cp


def set_default_spec(spec):
    global _default_spec
    _default_spec = spec


class PartialSpec(DistSpec):
    spec = None

    def __init__(self, *args, **kwargs):
        super(PartialSpec, self).__init__(self.spec, *args, **kwargs)


class UniformSpec(PartialSpec):
    spec = pm.Uniform
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, lower=0, upper=1):
        super(UniformSpec, self).__init__(lower=lower, upper=upper)


class NormalSpec(PartialSpec):
    spec = pm.Normal
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, mu=0, sd=1):
        super(NormalSpec, self).__init__(mu=mu, sd=sd)


class BetaSpec(PartialSpec):
    spec = pm.Beta
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, alpha=1, beta=1):
        super(BetaSpec, self).__init__(alpha=alpha, beta=beta)


class ExponentialSpec(PartialSpec):
    spec = pm.Exponential
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, lam=1):
        super(ExponentialSpec, self).__init__(lam=lam)


class LaplaceSpec(PartialSpec):
    spec = pm.Laplace
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, mu=0, b=1):
        super(LaplaceSpec, self).__init__(mu=mu, b=b)


class StudentTSpec(PartialSpec):
    spec = pm.StudentT
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, nu, mu=0, sd=1):
        super(StudentTSpec, self).__init__(nu=nu, mu=mu, sd=sd)


class CauchySpec(PartialSpec):
    spec = pm.Cauchy
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, alpha=0, beta=1):
        super(CauchySpec, self).__init__(alpha=alpha, beta=beta)


class HalfCauchySpec(PartialSpec):
    spec = pm.HalfCauchy
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, beta):
        super(HalfCauchySpec, self).__init__(beta=beta)


class GammaSpec(PartialSpec):
    spec = pm.Gamma
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, alpha, beta):
        super(GammaSpec, self).__init__(alpha=alpha, beta=beta)


class WeibullSpec(PartialSpec):
    spec = pm.Weibull
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, alpha, beta):
        super(WeibullSpec, self).__init__(alpha=alpha, beta=beta)


class LognormalSpec(PartialSpec):
    spec = pm.Lognormal
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, mu=0, sd=1):
        super(LognormalSpec, self).__init__(mu=mu, sd=sd)


class ChiSquaredSpec(PartialSpec):
    spec = pm.ChiSquared
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, nu):
        super(ChiSquaredSpec, self).__init__(nu=nu)


class HalfNormalSpec(PartialSpec):
    spec = pm.HalfNormal
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, sd=1):
        super(HalfNormalSpec, self).__init__(sd=sd)


class WaldSpec(PartialSpec):
    spec = pm.Wald
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, mu, lam, alpha=0.):
        super(WaldSpec, self).__init__(mu=mu, lam=lam, alpha=alpha)


class ParetoSpec(PartialSpec):
    spec = pm.Pareto
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, alpha, m):
        super(ParetoSpec, self).__init__(alpha=alpha, m=m)


class InverseGammaSpec(PartialSpec):
    spec = pm.InverseGamma
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, alpha, beta=1):
        super(InverseGammaSpec, self).__init__(alpha=alpha, beta=beta)


class ExGaussianSpec(PartialSpec):
    spec = pm.ExGaussian
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, mu, sd, nu):
        super(ExGaussianSpec, self).__init__(mu=mu, sigma=sd, nu=nu)


class VonMisesSpec(PartialSpec):
    spec = pm.VonMises
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, mu, kappa):
        super(VonMisesSpec, self).__init__(mu=mu, kappa=kappa)


class SkewNormalSpec(PartialSpec):
    spec = pm.SkewNormal
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, mu=0.0, sd=1, alpha=1):
        super(SkewNormalSpec, self).__init__(mu=mu, sd=sd, alpha=alpha)

'''
class HalfStudentTSpec(PartialSpec):
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=pm.HalfStudentT.distribution.__name__,
        doc="""Bounded StudentT with support on [0, +inf]\n{doc}""".format(
            doc=pm.StudentT.__doc__
        )
    )
    spec = pm.HalfStudentT

    def __init__(self, nu, mu=0, sd=1):
        super(HalfStudentTSpec, self).__init__(nu=nu, mu=mu, sd=sd)
'''

'''
class NormalMixtureSpec(PartialSpec):
    spec = pm.NormalMixture
    __doc__ = """Gelato DistSpec with {dist} prior\n\n{doc}""".format(
        dist=spec.__name__,
        doc=spec.__doc__
    )

    def __init__(self, w, mu, sd=None, tau=None):
        w = np.asarray(w)
        mu = np.asarray(mu)
        if sd is not None:
            sd = np.asarray(sd)
        if tau is not None:
            tau = np.asarray(tau)
        _, sd = get_tau_sd(tau, sd)
        super(NormalMixtureSpec, self).__init__(w=w, mu=mu, sd=sd)
'''
