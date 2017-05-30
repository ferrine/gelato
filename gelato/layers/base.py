import functools
import inspect
import six
import lasagne.layers.base
import pymc3 as pm
from pymc3.memoize import hashable

from gelato.specs.dist import get_default_spec, FlatSpec
from gelato.specs.base import DistSpec

__all__ = [
    'LayerModelMeta',
    'Layer',
    'MergeLayer',
    'bayes'
]


class LayerModelMeta(pm.model.InitContextMeta):
    """Magic comes here
    """

    def __init__(cls, what, bases, dic):
        from gelato.layers.helper import find_parent
        super(LayerModelMeta, cls).__init__(what, bases, dic)
        # make flexible property for new class

        def fget(self):
            if self._name is None:
                return '{}_{}'.format(self.__class__.__name__, self._fingerprint)
            else:
                return self._name

        def fset(self, value):
            if not value:
                self._name = None
            else:
                self._name = str(value)

        cls._name = None
        cls.name = property(fget, fset)

        # wrap init for new class
        def wrap_init(__init__):
            @functools.wraps(__init__)
            def wrapped(self, *args, **kwargs):
                name = kwargs.get('name')
                self._fingerprint = hashable(self.parent)
                pm.Model.__init__(self, name)
                __init__(self, *args, **kwargs)
            return wrapped

        # wrap new for new class
        def wrap_new(__new__):
            @functools.wraps(__new__)
            def wrapped(_cls_, *args, **kwargs):
                parent = kwargs.get('model', None)
                if parent is None and not issubclass(_cls_, lasagne.layers.InputLayer):
                    incoming = kwargs.get('incoming',
                                          kwargs.get('incomings',
                                                     args[1]))
                    parent = find_parent(incoming)
                kwargs['model'] = parent
                instance = __new__(_cls_, *args, **kwargs)
                return instance
            return classmethod(wrapped)

        cls.__init__ = wrap_init(cls.__init__)
        cls.__new__ = wrap_new(cls.__new__)

        def add_param(self, spec, shape, name=None, **tags):
            if tags.get('trainable', True):
                if tags.get('regularizable', True):
                    if not isinstance(spec, DistSpec):
                        # here spec is like test value
                        # passed to pymc3 distribution
                        spec = getattr(self, 'default_spec', get_default_spec(spec))
                else:
                    spec = FlatSpec()
                if name is not None:
                    spec = spec.with_name(name)
            return lasagne.layers.base.Layer.add_param(
                self, spec, shape, **tags)
        cls.add_param = add_param

        # needed for working with lasagne tools
        def wrap_getitem(__getitem__):
            @functools.wraps(__getitem__)
            def wrapped(self, item):
                if not isinstance(item, six.string_types):
                    raise TypeError('%r object accepts only string keys'
                                    % self.__class__)
                else:
                    __getitem__(self, item)
            return wrapped

        cls.__getitem__ = wrap_getitem(cls.__getitem__)

    def __repr__(self):
        return '{}.{}'.format(self.__module__, self.__name__)

    @classmethod
    def __subclasshook__(cls, C):
        if lasagne.layers.Layer in C.__mro__ or pm.Model in C.__mro__:
            return True
        else:
            return False


def bayes(layercls, stack=1):
    try:
        issubcls = issubclass(layercls, lasagne.layers.base.Layer)
    except TypeError:
        raise TypeError('{} needs to be a Layer subclass'
                        .format(layercls))
    if issubcls:
        if type(layercls) is LayerModelMeta:
            raise TypeError('{} is already bayesian'
                            .format(layercls))
        else:
            @six.add_metaclass(LayerModelMeta)
            class BayesianAnalog(layercls, pm.Model):
                pass
            frm = inspect.stack()[stack]
            mod = inspect.getmodule(frm[0])
            if mod is None:
                modname = '__main__'
            else:
                modname = mod.__name__
            BayesianAnalog.__module__ = modname
            BayesianAnalog.__doc__ = layercls.__doc__
            BayesianAnalog.__name__ = layercls.__name__
            return BayesianAnalog
    else:
        raise TypeError('{} needs to be a Layer subclass'
                        .format(layercls))

Layer = bayes(lasagne.layers.base.Layer)
MergeLayer = bayes(lasagne.layers.base.MergeLayer)
