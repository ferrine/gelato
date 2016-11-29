from numpy import math
import numpy as np
from theano import tensor as tt

c = - 0.5 * math.log(2 * math.pi)


def kl_divergence_normal_pair(mu1, mu2, sd1, sd2):
    if not any([
        isinstance(mu1, tt.Variable),
        isinstance(mu2, tt.Variable),
        isinstance(sd1, tt.Variable),
        isinstance(sd2, tt.Variable),
                ]):
        elemwise_kl = (math.log(sd2/sd1) +
                       (sd2**2 + (mu1-mu2)**2)/(2.*sd2**2) -
                       0.5)   # type: np.ndarray
        return np.sum(elemwise_kl)
    else:
        elemwise_kl = (tt.log(sd2/sd1) +
                       (sd2**2 + (mu1-mu2)**2)/(2.*sd2**2) -
                       0.5)
        return tt.sum(elemwise_kl)


def log_normal(x, mean, std, eps=0.0):
    std += eps
    return c - tt.log(tt.abs_(std)) - (x - mean) ** 2 / (2 * std ** 2)


def log_normal3(x, mean, rho, eps=0.0):
    std = ttrho2sd(rho)
    return log_normal(x, mean, std, eps)


def npsd2rho(sd):
    """sd -> rho
    numpy converter
    mu + sd*e = mu + log(1+exp(rho))*e"""
    return np.math.log(np.math.exp(sd) - 1)


def nptho2sd(rho):
    """rho -> sd
    numpy converter
    mu + sd*e = mu + log(1+exp(rho))*e"""
    return np.math.log1p(np.math.exp(rho))


def ttsd2rho(sd):
    """sd -> rho
    theano converter
    mu + sd*e = mu + log(1+exp(rho))*e"""
    return tt.log(tt.exp(sd) - 1)


def ttrho2sd(rho):
    """rho -> sd
    theano converter
    mu + sd*e = mu + log(1+exp(rho))*e"""
    return tt.log1p(tt.exp(rho))


def sd2rho(sd):
    """sd -> rho
    auto converter
    mu + sd*e = mu + log(1+exp(rho))*e"""
    if isinstance(sd, tt.Variable):
        return ttsd2rho(sd)
    else:
        return npsd2rho(sd)


def rho2sd(rho):
    """rho -> sd
    auto converter
    mu + sd*e = mu + log(1+exp(rho))*e"""
    if isinstance(rho, tt.Variable):
        return ttrho2sd(rho)
    else:
        return nptho2sd(rho)


def flatten(tensors):
    joined = tt.concatenate([var.ravel() for var in tensors])
    return joined
