import numpy as np
import theano.tensor as tt


def npsd2rho(sd):
    return np.math.log(np.math.exp(sd) - 1)


def nptho2sd(rho):
    return np.math.log1p(np.math.exp(rho))


def ttsd2rho(sd):
    return tt.log(tt.exp(sd) - 1)


def ttrho2sd(rho):
    return tt.log1p(tt.exp(rho))
