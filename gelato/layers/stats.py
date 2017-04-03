from .base import Layer


class ReplaceLayer(Layer):
    def get_output_for(self, incoming, approx=None, deterministic=False, **kwargs):
        return approx.apply_replacements(incoming, deterministic=deterministic)


class SamplingLayer(Layer):
    def get_output_for(self, incoming, approx=None, samples=10, **kwargs):
        return approx.sample_node(incoming, samples=samples)

    def get_output_shape_for(self, input_shape):
        return (None, ) + input_shape
