from .base import Layer


class PosteriorLayer(Layer):
    def get_output_for(self, incoming, approx=None, deterministic=False, **kwargs):
        return approx.apply_replacements(incoming, deterministic=deterministic)


class SamplingLayer(Layer):
    def __init__(self, incoming, samples=10, name=None):
        super(SamplingLayer, self).__init__(incoming, name=name)
        self.samples = samples

    def get_output_for(self, incoming, approx=None, **kwargs):
        return approx.sample_node(incoming, size=self.samples)

    def get_output_shape_for(self, input_shape):
        return (None, ) + input_shape
