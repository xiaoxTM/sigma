import tensorflow as tf
import numpy as np
from .. import colors
from ..ops import helper

def flatten(input_shape, name=None):
    if name is None:
        name = helper.dispatch_name('flatten')
    output_shape = [-1, np.prod(input_shape[1:])]
    scope = tf.name_scope(name)
    def _flatten(x):
        with scope:
            return tf.reshape(x, output_shape, name)
    return _flatten, output_shape


def reshape(output_shape, name=None):
    if name is None:
        name = helper.dispatch_name('reshape')
    scope = tf.name_scope(name)
    def _reshape(x):
        with scope:
            return tf.reshape(x, output_shape, name)
    return _reshape, output_shape
