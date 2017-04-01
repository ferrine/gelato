import pytest
import theano
import pymc3 as pm
import numpy as np
import lasagne.updates as updates
import lasagne.nonlinearities as to
from gelato.layers import DenseLayer, InputLayer
from gelato.variational.elbo import sample_elbo
from gelato.layers.helper import get_output
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
        with pm.Model() as model:
            inp = InputLayer(self.x.shape, input_var=input_var)
            out = DenseLayer(inp, 1, W=NormalSpec(sd=LognormalSpec()), nonlinearity=to.identity)
            pm.Normal('y', mu=get_output(out),
                      sd=self.sd,
                      observed=self.y)
        elbos, upd_rng, vp = sample_elbo(model, samples=1)
        upd_adam = updates.adagrad(-elbos.mean(), vp.params)
        upd_rng.update(upd_adam)
        step = theano.function([], elbos.mean(), updates=upd_rng)
        for i in range(1000):
            step()
        with pytest.raises(ValueError):
            get_output(out, deterministic=True)
        preds = get_output(out, vp=vp, deterministic=True)
        np.testing.assert_allclose(preds.eval(), self.y, rtol=0, atol=1)

