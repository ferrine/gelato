import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams
from gelato.random import set_np_rng, set_tt_rng
set_np_rng(numpy.random.RandomState(1))
set_tt_rng(MRG_RandomStreams(1))
