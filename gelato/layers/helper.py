import lasagne.layers.helper as _helper
from ..variational.utils import apply_replacements

__all__ = [
    'get_output'
]


def get_output(layer_or_layers, inputs=None,
               vp=None, deterministic=False, **kwargs):
    """Wrapper over lasagne helper function for user friendly
    replacements in the graph when getting output

    Parameters
    ----------
    layer_or_layers : Layer or list
        the :class:`Layer` instance for which to compute the output
        expressions, or a list of :class:`Layer` instances.

    inputs : None, Theano expression, numpy array, or dict
        If None, uses the input variables associated with the
        :class:`InputLayer` instances.
        If a Theano expression, this defines the input for a single
        :class:`InputLayer` instance. Will throw a ValueError if there
        are multiple :class:`InputLayer` instances.
        If a numpy array, this will be wrapped as a Theano constant
        and used just like a Theano expression.
        If a dictionary, any :class:`Layer` instance (including the
        input layers) can be mapped to a Theano expression or numpy
        array to use instead of its regular output.

    vp : gelato.variational.utils.VariationalParams - holder
        for variational params, used in replacements

    deterministic : bool - whether do deterministic replacements

    Returns
    -------
    output : Theano expression or list
        the output of the given layer(s) for the given network input
    """
    if deterministic and vp is None:
        raise ValueError('Cannot use deterministic argument '
                         'without variational params')
    output = _helper.get_output(
        layer_or_layers=layer_or_layers,
        inputs=inputs, **kwargs
    )
    if vp is None:
        return output
    elif not isinstance(output, list):
        return apply_replacements(output, vp, deterministic)
    else:
        return [
            apply_replacements(out, vp, deterministic)
            for out in output
        ]
