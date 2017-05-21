try:
    import pymc3
    if not hasattr(pymc3.Model, 'root'):
        raise ImportError('Need pymc3>=3.1')
    if not hasattr(pymc3.variational, 'opvi'):
        raise ImportError('Need latest pymc3 with OPVI')
except ImportError:
    raise ImportError('Need pymc3>=3.1')
else:
    del pymc3
    # for convenience
    from pymc3.theanof import set_tt_rng, tt_rng
    from gelato.specs import *
    from . import layers
    from . import specs
    from .version import __version__
