from sigma import colors

__backend__ = 'tensorflow'

print('Using {}{}{} backend'.format(colors.fg.red, __backend__, colors.reset))
if __backend__ == 'tensorflow':
    from .__tensorflow__ import *
elif __backend__ == 'theano':
    from .__theano__ import *
else:
    raise ValueError('core of `{}` for sigma is not supported'
                     .format(__backend__))
