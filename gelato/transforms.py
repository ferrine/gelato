import numpy as np
import theano.tensor as tt


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
