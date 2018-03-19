from sigma import colors
from .commons import *

__backend__ = 'tensorflow'

print('Using {} backend'.format(colors.red(__backend__)))
if __backend__ == 'tensorflow':
    from .__tensorflow__ import *
elif __backend__ == 'theano':
    from .__theano__ import *
else:
    raise ValueError('core of `{}` for sigma is not supported'
                     .format(__backend__))
