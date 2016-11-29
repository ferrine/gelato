import numpy as np
from ..random import get_rng


def generate_data(intercept, slope, sd=.2, size=700):
    x = np.linspace(-10, 10, size)
    y = intercept + x * slope
    return x, y + get_rng().normal(size=size, scale=sd)
