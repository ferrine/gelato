from pymc3.model import modelcontext
from pymc3.distributions.distribution import Distribution
from functools import partial


class DistSpec(object):
    def __init__(self, distcls, *args, **kwargs):
        if not issubclass(distcls, Distribution):
            raise ValueError('We can deal with pymc3 distributions only')
        self.args = args
        self.kwargs = kwargs
        self.distcls = distcls

    def __call__(self, shape, name=None):
        model = modelcontext(None)
        if name is None:
            name = 'w{}'.format(len(model.vars))
        val = model.Var(
                name, self.distcls.dist(
                    *self.args, shape=shape, **self.kwargs
                ),
            )
        val.tag.test_value = val.random()
        return val

    def with_name(self, name):
        return partial(self, name=name)
