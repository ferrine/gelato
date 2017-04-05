import functools
import inspect
import pymc3 as pm
from theano import theano
from theano.tensor.basic import _tensor_py_operators

from lasagne import init
from pymc3.distributions import distribution as dist
from gelato._compile import define


class BaseSpec(init.Initializer):
    def with_name(self, name):
        return functools.partial(self, name=name)

    @classmethod
    def _call_args(cls, args, name, shape):
        return [
            cls._call(arg, '{}.{}'.format(name, i), shape)
            for i, arg in enumerate(args)
        ]

    @classmethod
    def _call_kwargs(cls, kwargs, name, shape):
        return {
            key: cls._call(arg, '{}:{}'.format(name, key), shape)
            for key, arg in kwargs.items()
        }

    @staticmethod
    def _call(arg, label, shape):
        if isinstance(arg, BaseSpec):
            return arg(shape, label)
        elif isinstance(arg, init.Initializer):
            return arg(shape)
        else:
            return arg

    def __call__(self, shape, name=None):
        raise NotImplementedError


class SpecOp(BaseSpec):
    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def __call__(self, shape, name=None):
        if name is not None:
            name += self.op.__name__
        args = self._call_args(self.args, name, shape)
        kwargs = self._call_kwargs(self.kwargs, name, shape)
        return self.op(*args, **kwargs)

    def __repr__(self):
        return self.op.__name__

    __str__ = __repr__


head = """\
class _spec_py_operators(object):
"""
mth_template = """\
    def {0}{signature}:
        return SpecOp{inner_signature}
"""
meths = []
globs = dict(SpecOp=SpecOp)
for key, mth in _tensor_py_operators.__dict__.items():
    if callable(mth):
        argspec = inspect.getfullargspec(mth)
        signature = inspect.formatargspec(*argspec)
        inner_signature = inspect.formatargspec(
            args=['mth{0}'.format(key)] + argspec.args,
            varargs=argspec.varargs,
            varkw=argspec.varkw,
            defaults=argspec.args[1:],
            formatvalue=lambda value: '=' + str(value)
        )
        meths.append(mth_template.format(key, signature=signature, inner_signature=inner_signature))
        globs['mth{0}'.format(key)] = mth

_spec_py_operators = define('_spec_py_operators', head + '\n'.join(meths), globs, 1)


class DistSpec(BaseSpec, _spec_py_operators):
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
        try:
            valid_cls = issubclass(distcls, dist.Distribution)
        except TypeError:
            valid_cls = False
        try:
            valid_inst = isinstance(distcls, pm.Bound)
        except ValueError:
            valid_inst = False
        if not (valid_cls or valid_inst):
            raise ValueError('We can deal with pymc3 '
                             'distributions only, got {!r} instead'
                             .format(distcls))
        self.testval = kwargs.pop('testval', None)
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
                    **called_kwargs,
                    dtype=theano.config.floatX
                ),
            )
        if self.testval is None:
            val.tag.test_value = get_default_testval()(shape).astype(val.dtype)
        elif isinstance(self.testval, str) and self.testval == 'random':
            val.tag.test_value = val.random(size=shape).astype(val.dtype)
        else:
            val.tag.test_value = self.testval(shape).astype(val.dtype)
        return val

    def __repr__(self):
        template = '<{cls}: {args!r}; {kwargs!r}>'
        return template.format(cls=self.distcls.__name__,
                               args=self.args,
                               kwargs=self.kwargs)


_default_testval = init.GlorotUniform()


def set_default_testval(testval):
    global _default_testval
    _default_testval = testval


def get_default_testval():
    return _default_testval
