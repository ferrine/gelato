import six
import functools
import inspect
import pymc3 as pm
import lasagne.layers.base
from ..spec import DistSpec, get_default_spec

__all__ = [
    'LayerModelMeta',
    'Layer',
    'MergeLayer',
    'bayes'
]


class LayerModelMeta(pm.model.InitContextMeta):
    """Magic comes here
    """
    def __new__(mcs, what, bases, dic):
        # add model to bases to get all features from pymc3
        bases = tuple(list(bases) + [pm.Model])
        # create new class
        newcls = super(LayerModelMeta, mcs).__new__(mcs, what, bases, dic)
        return newcls

    def __init__(cls, what, bases, dic):
        from gelato.layers.helper import find_parent
        super(LayerModelMeta, cls).__init__(what, bases, dic)
        # make flexible property for new class

        def fget(self):
            if self._name is None:
                return '{}_{}'.format(self.__class__.__name__, id(self))
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
                pm.Model.__init__(self, name)
                __init__(self, *args, **kwargs)
            return wrapped

        # wrap new for new class
        def wrap_new(__new__):
            @functools.wraps(__new__)
            def wrapped(_cls_, *args, **kwargs):
                parent = kwargs.get('model', None)
                if parent is None:
                    parent = find_parent(args[0])
                kwargs['model'] = parent
                instance = __new__(_cls_, *args, **kwargs)
                return instance
            return classmethod(wrapped)

        cls.__init__ = wrap_init(cls.__init__)
        cls.__new__ = wrap_new(cls.__new__)

        def add_param(self, spec, shape, name=None, **tags):
            if not isinstance(spec, DistSpec):
                spec = getattr(self, 'default_spec', get_default_spec())
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


def bayes(layercls):
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
            class BayesianAnalog(
                    six.with_metaclass(LayerModelMeta, layercls)
                    ):
                pass
            frm = inspect.stack()[1]
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
