import theano.tensor as tt
import theano
from .utils import variational_replacements, flatten
from .math import log_normal3


def sample_elbo(model, population=None, samples=1, pi=1):
    """ pi*KL[q(w|mu,rho)||p(w)] + E_q[log p(D|w)]
    approximated by Monte Carlo sampling

    Parameters
    ----------
    model : pymc3.Model
    population : dict - maps observed_RV to its population size
        if not provided defaults to full population
    samples : number of Monte Carlo samples used for approximation,
        defaults to 1
    pi : additional coefficient for KL[q(w|mu,rho)||p(w)] as proposed in [1]_

    Returns
    -------
    (E_q[elbo], V_q[elbo], updates, SharedADVIFit)
        mean, variance of elbo, updates for random streams, shared dicts

    Notes
    -----
    You can pass tensors for `pi`  and `samples` to control them while
        training

    References
    ----------
    .. [1] Charles Blundell et al: "Weight Uncertainty in Neural Networks"
        arXiv preprint arXiv:1505.05424
    """
    if population is None:
        population = dict()
    replacements, _, shared = variational_replacements(model)
    x = flatten(replacements.values())
    mu = flatten(shared.means.values())
    rho = flatten(shared.rhos.values())

    def likelihood(var):
        tot = population.get(var, population.get(var.name))
        logpt = tt.sum(var.logpt)
        if tot is not None:
            tot = tt.as_tensor(tot)
            logpt *= tot / var.size
        return logpt

    log_p_D = tt.add(*map(likelihood, model.root.observed_RVs))
    log_p_W = model.root.varlogpt + tt.sum(model.root.potentials)
    log_q_W = tt.sum(log_normal3(x, mu, rho))
    _elbo_ = log_p_D + pi * (log_p_W - log_q_W)
    _elbo_ = theano.clone(_elbo_, replacements, strict=False)

    samples = tt.as_tensor(samples)
    elbos, updates = theano.scan(fn=lambda: _elbo_,
                                 outputs_info=None,
                                 n_steps=samples)
    return tt.mean(elbos), tt.var(elbos), updates, shared
