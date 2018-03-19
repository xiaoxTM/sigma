from . import core

def get(optimizer, **kwargs):
    """ get optimizer from None | string | Tensor | callable function
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
            raise NotImplementedError('optimizer `{}` not implemented'
                                      .format(optimizer))
        return eval('tf.train.{}(**kwargs)'.format(optimizer))
    elif isinstance(optimizer, core.Optimizer) or callable(optimizer):
        return optimizer
    else:
        raise ValueError('cannot get optimizer `{}` with type {}'
                         .format(optimizer, type(optimizer)))
