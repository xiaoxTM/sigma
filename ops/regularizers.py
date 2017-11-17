import tensorflow as tf

def regularize(l1=0.0, l2=0.0):
    def _regularize(x):
        regularization = 0.
        if l1:
            regularization += tf.reduce_sum(l1 * tf.abs(x))
        if l2:
            regularization += tf.reduce_sum(l2 * tf.square(x))
        return regularization

    return _regularize

def get(regularizer):
    """ get regularizers from None | string | callable function
    """
    if regularizer is None:
        return None
    elif isinstance(regularizer, str):
        """
        // TODO: introduce regular expression (import re module)
        splits should like ['l1=xxx', 'l2=xxx']
        """
        splits = regularizer.split(',')
        return eval('regularize({}, {})'.format(splits[0], splits[1]))
    elif callable(regularizer):
        return regularizer
    else:
        raise ValueError('cannot get regularizer `{}` with type {}'
                         .format(regulaizer, type(regulaizer)))
