from theano import theano
from .datasets import generate_linear_regression
from gelato.layers import *
import gelato
import pymc3 as pm


def test_stats_layers():
    intercept = .2
    slope = 2
    x_, _ = generate_linear_regression(intercept, slope)
    x = theano.shared(x_[None].T)
    inp = InputLayer((None, 1), input_var=x)
    nnet = DenseLayer(inp, 3)
    nnet = DenseLayer(nnet, 2)
    nnet_post = PosteriorLayer(nnet)
    nnet_sample = SamplingLayer(nnet, samples=10)
    with nnet_post.root:
        approx = pm.MeanField()
        out_ = get_output(nnet_post, approx=approx)
        out = out_.eval()  # should work
        assert out.shape == (x_.shape[0], 2)
    with nnet_sample.root:
        approx = pm.MeanField()
        out_ = get_output(nnet_sample, approx=approx)
        out = out_.eval()  # should work
        assert out.shape == (10, x_.shape[0], 2)


def test_normalizaton():
    inp = InputLayer((None, 1))
    nnet = BatchNormLayer(inp)
    assert isinstance(nnet.mean, theano.compile.SharedVariable)
    assert isinstance(nnet.inv_std, theano.compile.SharedVariable)
    assert isinstance(nnet.gamma.distribution, gelato.get_default_spec().distcls)
    assert isinstance(nnet.beta.distribution, pm.Flat)
    y_l = DenseLayer(nnet, 2)
    with y_l.root:
        x, y = generate_linear_regression(1, 1)
        y_ = get_output(y_l, x[:, None].astype('float32'))
        pm.Normal('y', y_[:, 0], theano.tensor.exp(y_[:, 1]), observed=y.astype('float32'))
        pm.fit(10)
