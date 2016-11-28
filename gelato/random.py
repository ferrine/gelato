"""
A module with a package-wide random number generator,
used for weight initialization and seeding noise layers.
This can be replaced by a :class:`numpy.random.RandomState` instance with a
particular seed to facilitate reproducibility.
"""

import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams


_np_rng = np.random
_tt_rng = MRG_RandomStreams()


def np_rng():
    """Get the package-level random number generator.
    Returns
    -------
    :class:`numpy.random.RandomState` instance
        The :class:`numpy.random.RandomState` instance passed to the most
        recent call of :func:`set_np_rng`, or ``numpy.random``
        if :func:`set_np_rng` has never been called.
    """
    return _np_rng

# alias as in lasagne
get_rng = np_rng


def tt_rng():
    """Get the package-level random number generator.
    Returns
    -------
    :class:`theano.sandbox.rng_mrg.MRG_RandomStreams` instance
        The :class:`theano.sandbox.rng_mrg.MRG_RandomStreams`
        instance passed to the most recent call of :func:`set_tt_rng`
    """
    return _tt_rng


def set_np_rng(new_rng):
    """Set the package-level random number generator.
    Parameters
    ----------
    new_rng : ``numpy.random`` or a :class:`numpy.random.RandomState` instance
        The random number generator to use.
    """
    global _np_rng
    _np_rng = new_rng

# alias as in lasagne
set_rng = set_np_rng


def set_tt_rng(new_rng):
    """Set the package-level random number generator.
    Parameters
    ----------
    new_rng : :class:`theano.sandbox.rng_mrg.MRG_RandomStreams` instance
        The random number generator to use.
    """
    global _tt_rng
    _tt_rng = new_rng



