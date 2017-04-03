import pytest
import theano
import pymc3 as pm
import numpy as np
import lasagne.nonlinearities as to
from gelato.layers import DenseLayer, InputLayer
from gelato.layers import get_output
from .datasets import generate_data
from gelato.spec import NormalSpec, LognormalSpec


class TestWorkflow(object):
    @classmethod
    def setup_class(cls):
        cls.intercept = 1
        cls.slope = 3
        cls.sd = .1
        cls.x, cls.y = generate_data(cls.intercept, cls.slope, sd=cls.sd)
        cls.x = np.matrix(cls.x).T
        cls.y = np.matrix(cls.y).T

    def test_workflow(self):
        input_var = theano.shared(self.x)
        inp = InputLayer(self.x.shape, input_var=input_var)
        out = DenseLayer(inp, 1, W=NormalSpec(sd=LognormalSpec()), nonlinearity=to.identity)
        out = DenseLayer(out, 1, W=NormalSpec(sd=LognormalSpec()), nonlinearity=to.identity)
        assert out.root is inp
        with out:
            pm.Normal('y', mu=get_output(out),
                      sd=self.sd,
                      observed=self.y)

