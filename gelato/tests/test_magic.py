import unittest
import theano
import pymc3 as pm
import numpy as np
import lasagne.layers as layers
import lasagne.updates as updates
import lasagne.nonlinearities as to
from gelato.layers.base import Layer
from gelato.spec import DistSpec
from gelato.variational.elbo import sample_elbo
from gelato.layers.helper import get_output
from .datasets import generate_data


class BayesDenseLayer(layers.DenseLayer, Layer):
    default_spec = DistSpec(pm.Normal, sd=DistSpec(pm.Lognormal, sd=1))


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.intercept = 1
        self.slope = 3
        self.sd = .1
        self.x, self.y = generate_data(self.intercept, self.slope, sd=self.sd)
        self.x = np.matrix(self.x).T
        self.y = np.matrix(self.y).T

    def test_workflow(self):
        input_var = theano.shared(self.x)
        with pm.Model() as model:
            inp = layers.InputLayer(self.x.shape, input_var=input_var)
            out = BayesDenseLayer(inp, 1, nonlinearity=to.identity)
            pm.Normal('y', mu=get_output(out),
                      sd=self.sd,
                      observed=self.y)
        elbo, _, upd_rng, vp = sample_elbo(model, samples=1)
        upd_adam = updates.adagrad(-elbo, vp.params)
        upd_rng.update(upd_adam)
        step = theano.function([], elbo, updates=upd_rng)
        for i in range(100):
            step()
        self.assertRaises(ValueError, get_output, out, deterministic=True)
        preds = get_output(out, vp=vp, deterministic=True)
        np.testing.assert_allclose(preds.eval(), self.y, rtol=0, atol=1)
