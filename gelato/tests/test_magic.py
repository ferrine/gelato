import lasagne.nonlinearities as to
import numpy as np
import pymc3 as pm
from lasagne import layers as llayers

from gelato.layers import DenseLayer, InputLayer
from gelato.layers import get_output, find_parent, find_root
from gelato.specs.dist import NormalSpec, LognormalSpec
from .datasets import generate_data


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
        inp = InputLayer(self.x.shape)
        out = DenseLayer(inp, 1, W=NormalSpec(sd=LognormalSpec()), nonlinearity=to.identity)
        out = DenseLayer(out, 1, W=NormalSpec(sd=LognormalSpec()), nonlinearity=to.identity)
        assert out.root is inp
        with out:
            pm.Normal('y', mu=get_output(out),
                      sd=self.sd,
                      observed=self.y)

    def test_find_parent_and_root(self):
        inp = InputLayer(self.x.shape)
        middle = llayers.DenseLayer(inp, 10)
        middle1 = DenseLayer(middle, 10)
        middle2 = llayers.DenseLayer(middle1, 4)

        assert find_parent(middle2) is middle1
        assert find_root(middle2) is inp
