import itertools
import functools
import inspect
import pymc3 as pm
from theano import theano
from theano.tensor.basic import _tensor_py_operators

from lasagne import init
from pymc3.distributions import distribution as dist
from gelato._compile import define

__all__ = [
    'BaseSpec',
    'as_spec_op',
    'get_default_testval',
    'set_default_testval',
    'DistSpec'
]


class BaseSpec(init.Initializer):
    memo = {}
    _counter = itertools.count(0)
    name = None

    def auto(self):
        if self.name is None:
            return 'auto_{}'.format(next(type(self)._counter))
        else:
            name = self.name
            self.name = None
            return name

    def with_name(self, name):
        self.name = name
        return self

    def _call_args(self, args, name, shape, memo):
        return [
            self._call(arg, '{}.{}'.format(name, i)
                       if name is not None and not name.startswith('auto')
                       else self.auto(), shape, memo)
            for i, arg in enumerate(args)
        ]

    def _call_kwargs(self, kwargs, name, shape, memo):
        return {
            key: self._call(arg, '{}:{}'.format(name, next(self._counter), memo)
                            if name is not None and not name.startswith('auto')
                            else self.auto(), shape, memo)
            for key, arg in kwargs.items()
        }

    @staticmethod
    def _call(arg, label, shape, memo):
        if isinstance(arg, BaseSpec):
            return arg(shape, label, memo)
        elif isinstance(arg, init.Initializer):
            return arg(shape)
        else:
            return arg

    def __call__(self, shape, name=None, memo=None):
        raise NotImplementedError


head = """\
class SpecVar(BaseSpec):
    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def __call__(self, shape, name=None, memo=None):
        if memo is None:
            memo = {}
        if name is None:
            name = self.auto()
        if id(self) in memo:
            return memo[id(self)]
        args = self._call_args(self.args, name, shape, memo)
        kwargs = self._call_kwargs(self.kwargs, name, shape, memo)
        memo[id(self)] = self.op(*args, **kwargs)
        return memo[id(self)]

    def __repr__(self):
        if hasattr(self.op, '__name__'):
            return 'SpecOp.' + self.op.__name__
        else:
            return 'SpecOp.' + type(self.op).__name__

    __str__ = __repr__

"""
mth_template = """\
    def {0}{signature}:
        return SpecVar{inner_signature}
"""
meths = []
globs = dict(BaseSpec=BaseSpec)
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

SpecVar = define('SpecVar', head + '\n'.join(meths), globs, 1)


def as_spec_op(func):
    return functools.partial(SpecVar, func)


class DistSpec(SpecVar):
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

    def __call__(self, shape, name=None, memo=None):
        if memo is None:
            memo = {}
        if name is None:
            name = self.auto()
        if id(self) in memo:
            return memo[id(self)]
        model = pm.modelcontext(None)
        called_args = self._call_args(self.args, name, shape, memo)
        called_kwargs = self._call_kwargs(self.kwargs, name, shape, memo)
        called_kwargs.update(shape=shape)
        val = model.Var(
                name, self.distcls.dist(
                    *called_args,
                    dtype=theano.config.floatX,
                    **called_kwargs
                ),
            )
        if self.testval is None:
            val.tag.test_value = get_default_testval()(shape).astype(val.dtype)
        elif isinstance(self.testval, str) and self.testval == 'random':
            val.tag.test_value = val.random(size=shape).astype(val.dtype)
        else:
            val.tag.test_value = self.testval(shape).astype(val.dtype)
        memo[id(self)] = val
        return memo[id(self)]

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
