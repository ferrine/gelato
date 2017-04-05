from lasagne.layers import InputLayer as _InputLayer
import numpy as np
from .base import bayes as _bayes
from theano.compile import SharedVariable

__all__ = [
    'InputLayer'
]


@_bayes
class InputLayer(_InputLayer):
    def __init__(self, shape, input_var=None, name=None, testval=None, **kwargs):
        _InputLayer.__init__(self, shape, input_var=input_var, name=name, **kwargs)
        if testval is not None:
            self.input_var.tag.test_value = testval
        if (not isinstance(self.input_var, SharedVariable)
           and not hasattr(self.input_var.tag, 'test_value')):
            shape = [s if s is not None else 2 for s in self.shape]
            dtype = self.input_var.dtype
            self.input_var.tag.test_value = np.random.uniform(size=shape).astype(dtype)
