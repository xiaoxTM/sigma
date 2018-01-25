import tensorflow as tf
import numpy as np
from .. import colors
from ..ops import helper


def flatten(input_shape,
            reuse=False,
            name=None,
            scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'flatten', reuse)
    output_shape = [-1, np.prod(input_shape[1:])]
    def _flatten(x):
        with ops_scope:
            return tf.reshape(x, output_shape, name)
    return _flatten, output_shape


def reshape(output_shape,
            reuse=False,
            name=None,
            scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'flatten', reuse)
    def _reshape(x):
        with ops_scope:
            return tf.reshape(x, output_shape, name)
    return _reshape, output_shape


def argmax(inputs,
           axis=None,
           dtype='int64',
           reuse=False,
           name=None,
           scope=None):
    return tf.argmax(inputs, axis, dtype, reuse, name, scope)


def argmin(inputs,
           axis=None,
           dtype='int64',
           reuse=False,
           name=None,
           scope=None):
    return tf.argmin(inputs, axis, dtype, reuse, name, scope)


def predict(predop=None,
            axis=None,
            dtype='int64',
            reuse=False,
            name=None,
            scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'flatten', reuse)
    if predop is None:
        predop = tf.argmax
    elif isinstance(predop, str):
        if predop == 'argmax':
            predop = tf.argmax
        elif predop == 'argmin':
            predop = tf.argmin
        else:
            raise ValueError('`predop` must be one of `argmax` or `argmin`.'
                             ' given {}'.format(predop))
    elif not callable(predop):
        raise TypeError('`predop` must be type of None or str or callable. '
                        'given {}'.format(type(predop)))
    def _predict(x):
        with ops_scope:
            return predop(x, axis, dtype, reuse, name, scope)
    return _predict
