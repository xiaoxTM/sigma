import tensorflow as tf
import numpy as np
from .. import status

def regularize(l1=0.0, l2=0.0):
    def _regularize(x):
        regularization = 0.
        if l1:
            regularization += tf.reduce_sum(l1 * tf.abs(x))
        if l2:
            regularization += tf.reduce_sum(l2 * tf.square(x))
        return regularization

    return _regularize


""" total variation regularization
    in case of x : [batch-size, rows, cols, depth, channels]
        return:
            r = mean([batch-size, 1:, cols, depth, channels] - [batch-size, :rows-1, cols, depth, channels])
            c = mean([batch-size, rows, 1:, depth, channels] - [batch-size, rows, :cols-1, depth, channels])
            d = mean([batch-size, rows, cols, 1:, channels] - [batch-size, rows, cols, :depth-1, channels])
            return 3 * (r + c + d)

    in case of x : [batch-size, rows, cols, channels]
        return:
            r = mean([batch-size, 1:, cols, channels] - [batch-size, :rows-1, cols, channels])
            c = mean([batch-size, rows, 1:, channels] - [batch-size, rows, :cols-1, channels])
            return 2 * (r + c)

    in case of x : [batch-size, neurons, channels]
        return mean([batch-size, 1:, channels] - [batch-size, :neurons-1, channels]) / (neurons)
"""
def total_variation_regularizer(shape):
    axes = list(range(len(shape)))
    del axes[status.axis]
    del axes[0]
    indices = [np.arange(s) for s in shape]

    def _differ(x, axis):
        previous = np.copy(indices)
        latter = np.copy(indices)
        previous[axis] = np.delete(previous[axis], -1) # re-indexing [0:-1]
        latter[axis] = np.delete(latter[axis], 0) # re-indexing [1:]
        previous = np.meshgrid(*previous, indexing='ij')
        previous = np.stack(previous, axis=-1)
        previous = tf.gather_nd(x, previous)
        latter = np.meshgrid(*latter, indexing='ij')
        latter = np.stack(latter, axis=-1)
        latter = tf.gather_nd(x, latter)
        differences = tf.nn.l2_loss(latter - previous)
        return tf.reduce_mean(differences)

    def _total_variation_regularizer(x):
        return np.sum([_differ(x, axis) for axis in axes])

    return _total_variation_regularizer


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
