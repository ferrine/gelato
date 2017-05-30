import numpy as np


def generate_linear_regression(intercept, slope, sd=.2, size=700):
    x = np.random.uniform(-10, 10, size)
    y = intercept + x * slope
    return x, y + np.random.normal(size=size, scale=sd)


def generate_sinus_regression(intercept, slope, sd=.2, size=700):
    x = np.random.uniform(-10, 10, size)
    y = intercept + (x + np.sin(x)) * slope
    return x, y + np.random.normal(size=size, scale=sd)
