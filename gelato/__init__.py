import pymc3
if not hasattr(pymc3.Model, 'root'):
    raise ImportError('Need pymc3>=3.1')
else:
    del pymc3
    from gelato.specs import dist
    from . import random
    from . import layers
    from .version import __version__
