import tensorflow as tf
import numpy as np
from .. import colors
from . import helper


def placeholder(dtype, shape=None, name=None):
    return tf.placeholder(dtype, shape, name)


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
