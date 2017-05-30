from .base import Layer

__all__ = [
    'PosteriorLayer',
    'SamplingLayer'
]


class PosteriorLayer(Layer):
    """
    Layer that makes replacements in the graph in the incoming layer 
    so that `get_output` can be evaluated 
    """
    def get_output_for(self, incoming, approx=None, deterministic=False, **kwargs):
        if approx is None:
            raise ValueError('No approximation specified')
        return approx.apply_replacements(incoming, deterministic=deterministic)


class SamplingLayer(Layer):
    """
    Layer that makes replacements in the graph and additionally samples incoming 
    layer.
    """
    def __init__(self, incoming, samples=10, name=None):
        super(SamplingLayer, self).__init__(incoming, name=name)
        self.samples = samples

    def get_output_for(self, incoming, approx=None, **kwargs):
        if approx is None:
            raise ValueError('No approximation specified')
        return approx.sample_node(incoming, size=self.samples)

    def get_output_shape_for(self, input_shape):
        return (None, ) + input_shape
