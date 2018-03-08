from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def get(optimizer, **kwargs):
    """ get optimizer from None | string | callable function
    """
    if optimizer is None:
        return None
    elif isinstance(optimizer, str):
        """
        // TODO: introduce regular expression (import re module)
        splits should like ['l1=xxx', 'l2=xxx']
        """
        if optimizer not in ['Optimizer', 'GradientDescentOptimizer',
                             'AdadeltaOptimizer', 'AdagradOptimizer',
                             'AdagradDAOptimizer', 'MomentumOptimizer',
                             'AdamOptimizer', 'FtrlOptimizer',
                             'ProximalGradientDescentOptimizer',
                             'ProximalAdagradOptimizer',
                             'RMSPropOptimizer']:
            raise NotImplementedError('optimizer `{}` not implemented'.format(optimizer))
        return eval('tf.train.{}(**kwargs)'.format(optimizer))
    elif callable(optimizer):
        return optimizer
    else:
        raise ValueError('cannot get optimizer `{}` with type {}'
                         .format(optimizer, type(optimizer)))