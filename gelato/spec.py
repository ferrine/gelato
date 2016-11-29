import pymc3 as pm
import pymc3.distributions.distribution
import functools


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
            raise ValueError('We can deal with pymc3 distributions only')
        self.args = args
        self.kwargs = kwargs
        self.distcls = distcls

    def __call__(self, shape, name=None):
        print(self)
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
