from theano import theano, tensor as tt
import pymc3 as pm
from .datasets import generate_sinus_regression
from gelato.layers import *


def test_posterior_layer():
    intercept = .2
    slope = 2
    x_, y_ = generate_sinus_regression(intercept, slope)
    x = theano.shared(x_[None].T)
    y = theano.shared(y_)
    inp = InputLayer((None, 1), input_var=x)
    nnet = DenseLayer(inp, 3)
    nnet = DenseLayer(nnet, 2)
    nnet = PosteriorLayer(nnet)
    with nnet.root:
        approx = pm.MeanField()
        out = get_output(nnet, approx=approx)
        mu, logsd = out[:, 0], out[:, 1]
        pm.Normal('lik', mu=mu.flatten(), sd=tt.exp(logsd).flatten(), observed=y)
        inference = pm.ADVI.from_mean_field(approx)
        inference.fit(10000)



