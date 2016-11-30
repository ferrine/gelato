import unittest
import numpy as np
import theano
from pymc3 import Model, Normal
from gelato.variational.elbo import sample_elbo
from gelato.variational.math import sd2rho


class TestElbo(unittest.TestCase):
    def test_elbo(self):
        mu0 = 1.5
        sigma = 1.0
        y_obs = np.array([1.6, 1.4])

        post_mu = 1.88
        post_sd = 1
        # Create a model for test
        with Model() as model:
            mu = Normal('mu', mu=mu0, sd=sigma)
            Normal('y', mu=mu, sd=1, observed=y_obs)

        # Create variational gradient tensor
        elbos, updates, vp = sample_elbo(model, samples=10000)

        vp.shared.means['mu'].set_value(post_mu)
        vp.shared.rhos['mu'].set_value(sd2rho(post_sd))

        f = theano.function([], elbos, updates=updates)
        elbo_mc = f()

        # Exact value
        elbo_true = (-0.5 * (
            3 + 3 * post_mu**2 - 2 * (y_obs[0] + y_obs[1] + mu0) * post_mu +
            y_obs[0]**2 + y_obs[1]**2 + mu0**2 + 3 * np.log(2 * np.pi)) +
            0.5 * (np.log(2 * np.pi) + 1))
        var_true = 3.15
        np.testing.assert_allclose(elbo_mc.var(), var_true, rtol=0, atol=1e-1)
        np.testing.assert_allclose(elbo_mc.mean(), elbo_true, rtol=0, atol=1e-1)

if __name__ == '__main__':
    unittest.main()
