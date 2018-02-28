from ..engine import core

if sigma.backend == 'tensorflow':
    from .__tensorflow__ import *
elif sigma.backend == 'theano':
    from .__theano__ import *
else:
    raise ValueError('core of `{}` for sigma is not supported'
                     .format(sigma.backend))
