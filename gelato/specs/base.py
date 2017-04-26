import six
import copy
import itertools
import functools
import inspect
import pymc3 as pm
from theano import theano
from theano.tensor.basic import _tensor_py_operators

from lasagne import init
from gelato._compile import define

__all__ = [
    'BaseSpec',
    'as_spec_op',
    'get_default_testval',
    'set_default_testval',
    'DistSpec'
]


class BaseSpec(init.Initializer):
    _counter = itertools.count(0)
    name = None
    tag = 'default'
    _shape = None

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

    def with_tag(self, tag):
        self.tag = tag
        return self

    def with_shape(self, shape):
        if callable(shape):
            self._shape = shape
        elif shape is not None:
            self._shape = shape
            self.tag = 'custom'
        else:
            self._shape = None
            self.tag = 'default'
        return self

    @staticmethod
    def _prepare(memo, shape):
        if memo is None:
            memo = {}
            if not isinstance(shape, dict):
                shape = {'default': shape}
            elif 'default' not in shape:
                raise ValueError('default shape not specified,'
                                 'please provide it with `default`'
                                 'key in input shape dict')
        return memo, shape

    def _get_shape(self, shape):
        """
        :param shape: dict 
        :return: dict 
        """
        if callable(self._shape):
            new_shape, tag = self._shape(shape[self.tag]), self.tag
        elif self._shape is not None:
            new_shape, tag = self._shape, self.tag
        else:
            new_shape, tag = shape[self.tag], self.tag
        shape = shape.copy()
        shape.update(default=new_shape)
        return shape, tag

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
            if isinstance(shape, dict):
                return arg(shape['default'])
        else:
            return arg

    def __call__(self, shape, name=None, memo=None):
        raise NotImplementedError


head = '''\
class SpecVar(BaseSpec):
    """
    Base class that supports delayed tensor operations
    """
    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def __call__(self, shape, name=None, memo=None):
        memo, shape = self._prepare(memo, shape)
        shape, tag = self._get_shape(shape)
        if name is None:
            name = self.auto()
        if id(self) ^ hash(tag) in memo:
            return memo[id(self) ^ hash(tag)]
        args = self._call_args(self.args, name, shape, memo)
        kwargs = self._call_kwargs(self.kwargs, name, shape, memo)
        memo[id(self) ^ hash(tag)] = self.op(*args, **kwargs)
        return memo[id(self) ^ hash(tag)]

    def __repr__(self):
        if hasattr(self.op, '__name__'):
            return 'SpecOp.' + self.op.__name__
        else:
            return 'SpecOp.' + type(self.op).__name__

    __str__ = __repr__

    def clone(self):
        return copy.deepcopy(self)

    def __iter__(self):
        raise NotImplementedError
'''
exclude = {
    '__iter__'
}
mth_template = """\
    def {0}{signature}:
        '''{doc}'''
        return SpecVar{inner_signature}
"""
meths = []
globs = dict(BaseSpec=BaseSpec, copy=copy)
for key, mth in _tensor_py_operators.__dict__.items():
    if callable(mth):
        if six.PY3:
            argspec = inspect.getfullargspec(mth)
            keywords = argspec.varkw
        else:
            argspec = inspect.getargspec(mth)
            keywords = argspec.keywords
        signature = inspect.formatargspec(*argspec)
        inner_signature = inspect.formatargspec(
            args=['mth{0}'.format(key)] + argspec.args,
            varargs=argspec.varargs,
            varkw=keywords,
            defaults=argspec.args[1:],
            formatvalue=lambda value: '=' + str(value)
        )
        meths.append(mth_template.format(
            key, signature=signature, inner_signature=inner_signature, doc=mth.__doc__))
        globs['mth{0}'.format(key)] = mth

SpecVar = define('SpecVar', head + '\n'.join(meths), globs, 1)


def as_spec_op(func):
    return functools.partial(SpecVar, func)


class DistSpec(SpecVar):
    """Spec based on pymc3 distributions

    All specs support lazy evaluation, see Usage

    Parameters
    ----------
    distcls : pymc3.Distribution
    args : args for `distcls`
    kwargs : kwargs for `distcls`

    Usage
    -----
    >>> spec = DistSpec(Normal, mu=0, sd=DistSpec(Lognormal, 0, 1))
    >>> spec += (NormalSpec() + LaplaceSpec()) / 100 - NormalSpec()
    >>> with Model():
    ...     prior_expr = spec((10, 10), name='silly_prior')

    """
    def __init__(self, distcls, *args, **kwargs):
        if not isinstance(distcls, type) and issubclass(distcls, pm.Distribution):
            raise ValueError('We can deal with pymc3 '
                             'distributions only, got {!r} instead'
                             .format(distcls))
        self.testval = kwargs.pop('testval', None)
        self.tag = kwargs.get('tag', 'default')
        self.args = args
        self.kwargs = kwargs
        self.distcls = distcls

    def __call__(self, shape, name=None, memo=None):
        memo, shape = self._prepare(memo, shape)
        if name is None:
            name = self.auto()
        shape, tag = self._get_shape(shape)
        if id(self) ^ hash(tag) in memo:
            return memo[id(self) ^ hash(tag)]
        model = pm.modelcontext(None)
        called_args = self._call_args(self.args, name, shape, memo)
        called_kwargs = self._call_kwargs(self.kwargs, name, shape, memo)
        called_kwargs.update(shape=shape['default'])
        val = model.Var(
            name, self.distcls.dist(
                *called_args,
                dtype=theano.config.floatX,
                **called_kwargs
            ),
        )
        if self.testval is None:
            val.tag.test_value = get_default_testval()(shape['default']).astype(val.dtype)
        elif isinstance(self.testval, str) and self.testval == 'random':
            val.tag.test_value = val.random(size=shape['default']).astype(val.dtype)
        else:
            val.tag.test_value = self.testval(shape['default']).astype(val.dtype)
        memo[id(self) ^ hash(tag)] = val
        return memo[id(self) ^ hash(tag)]

    def __repr__(self):
        if self._shape != -1:
            sh = '; '+str(self._shape)
        else:
            sh = ''
        template = '<{cls}: {args!r}; {kwargs!r}'+sh+'>'
        return template.format(cls=self.distcls.__name__,
                               args=self.args,
                               kwargs=self.kwargs)


_default_testval = init.Normal(std=.01)


def set_default_testval(testval):
    global _default_testval
    _default_testval = testval


def get_default_testval():
    return _default_testval
