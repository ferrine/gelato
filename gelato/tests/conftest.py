import numpy as np
import gelato
import pytest


@pytest.fixture()
def seeded():
    gelato.set_tt_rng(42)
    np.random.seed(42)
