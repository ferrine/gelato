from collections import namedtuple, OrderedDict
import numpy as np
import theano
import theano.tensor as tt
from gelato.variational.math import npsd2rho, rho2sd
from gelato.random import tt_rng


SharedADVIFit = namedtuple('SharedADVIFit', 'means, rhos')


def random_node_from_numpy(mu, sd, name):
    """

    Parameters
    ----------
    mu : np.array
    sd : np.array
    name : str

    Returns
    -------
    tuple : (mu + log(1+exp(rho))*e, mu, rho)
        mu - shared variable
        rho - shared variable
    """
    rho = theano.shared(npsd2rho(sd),
                        name='{}_rho_shared'.format(name))
    mu = theano.shared(mu, name='{}_mu_shared'.format(name))
    e = tt_rng().normal(rho.shape)
    return mu + rho2sd(rho) * e, mu, rho, e


def random_node(mu):
    rho = theano.shared(np.ones(mu.tag.test_value.shape))
    mu = theano.shared(mu.tag.test_value,
                       name='{}_mu_shared'.format(mu.name))
    e = tt_rng().normal(rho.shape)
    return mu + rho2sd(rho) * e, mu, rho, e


def variational_replacements_from_advifit(model, advifit=None):
    """Util for getting variational replacements from advifit

    Parameters
    ----------
    model : pymc3.Model
    advifit : pymc3.variational.advi.ADVIfit

    Returns
    -------
    tuple : (replacements, epsilons, SharedADVIfit)
    """
    replacements = OrderedDict()
    means = OrderedDict()
    rhos = OrderedDict()
    es = OrderedDict()
    for var in model.root.vars:
        if var.name in advifit.means:
            v, mu, rho, e = random_node_from_numpy(
                advifit.means[var.name],
                advifit.stds[var.name],
                var.name
            )
            replacements[var] = v
            means[var.name] = mu
            rhos[var.name] = rho
            es[var.name] = es
        else:
            continue
    return replacements, es, SharedADVIFit(means, rhos)


def variational_replacements(model):
    """Util for getting variational replacements

    Parameters
    ----------
    model : pymc3.Model

    Returns
    -------
    tuple : (replacements, epsilons, SharedADVIfit)
    """
    replacements = OrderedDict()
    means = OrderedDict()
    rhos = OrderedDict()
    es = OrderedDict()
    for var in model.root.vars:
        v, mu, rho, e = random_node(var)
        replacements[var] = v
        means[var.name] = mu
        rhos[var.name] = rho
        es[var.name] = e
    return replacements, es, SharedADVIFit(means, rhos)


def refresh_epsilon(es):
    """Create replacement dictionary for epsilons
    Parameters
    ----------
    es

    Returns
    -------

    """
    es_replacement = OrderedDict()
    for e in es:
        es_replacement[e] = tt_rng().normal(e.shape)
    return es_replacement


def flatten(tensors):
    joined = tt.concatenate([var.ravel() for var in tensors])
    return joined
