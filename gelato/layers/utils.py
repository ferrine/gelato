import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from lasagne.layers.base import Layer, MergeLayer
from lasagne.layers.helper import get_output
from pymc3.theanof import make_shared_replacements
from gelato.transforms import npsd2rho, ttrho2sd


def islayersub(cls):
    return not issubclass(cls, MergeLayer) and issubclass(cls, Layer)


def ismergesub(cls):
    return issubclass(cls, MergeLayer)


def get_shared_vars(model):
    return make_shared_replacements([], model.root)


def graph_with_advi_replacements(layer, advifit):
    rnd = MRG_RandomStreams()

    def make_node(mu, sd, name):
        rho = theano.shared(npsd2rho(sd),
                            name='{}_rho_shared'.format(name))
        mu = theano.shared(mu, name='{}_mu_shared'.format(name))
        return mu + ttrho2sd(rho) * rnd.normal(rho.shape)
    replacements = {}
    for label, var in layer.root.vars.items():
        if label in advifit.means:
            replacements[var] = make_node(
                advifit.means[label],
                advifit.sds[label],
                label
            )
        else:
            continue
    output = get_output(layer)
    return theano.clone(output, replacements, strict=False)
