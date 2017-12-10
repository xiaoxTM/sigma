import tensorflow as tf
import numpy as np

""" Computes the number of input and output units for a weight shape.
    Args:
        shape: Integer shape tuple or TF tensor shape.
    Returns:
        A tuple of scalars (fan_in, fan_out).

    // NOTE: code borrowed from tensorflow (https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/init_ops.py)
"""
def get_fans(shape):
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


""" he normal initializer
    initialize with parameter:
        mean = 0
        stddev = sqrt(2 / fan_in)
    where fan_in is the number of inputs units in the weight
"""
def he_normal(seed=None, name=None):
    def _he_normal(x, dtype, partition_info=None):
        fan_in, _ = get_fans(x)
        stddev = tf.sqrt(2 / fan_in)
        return tf.random_normal(x, 0, stddev, dtype, seed, name)
    return _he_normal


""" he uniform initializer
    initialize within [-limit, limit]:
        limit = sqrt(6 / fan_in)
    where fan_in is the number of inputs units in the weight
"""
def he_uniform(seed=None, name=None):
    def _he_uniform(x, dtype, partition_info=None):
#        return init.he_uniform(x, dtype)
        fan_in, _ = get_fans(x)
        limit = tf.sqrt(6 / fan_in)
        return tf.random_uniform(x, -limit, limit, dtype, seed, name)
    return _he_uniform


""" glorot normal initializer
    initialize with parameter:
        mean = 0
        stddev = sqrt(2 / (fan_in + fan_out))
    where fan_in is the number of inputs units in the weight
          fan_out is the output units
"""
def glorot_normal(seed=None, name=None):
    def _glorot_normal(x, dtype, partition_info=None):
        fan_in, fan_out = get_fans(x)
        stddev = tf.sqrt(2 / (fan_in + fan_out))
        return tf.random_normal(x, 0, stddev, dtype, seed, name)
    return _glorot_normal


""" glorot uniform initializer
    initialize within [-limit, limit]:
        limit = sqrt(6 / (fan_in + fan_out))
    where fan_in is the number of inputs units in the weight
          fan_out is the output units
"""
def glorot_uniform(seed=None, name=None):
    def _glorot_uniform(x, dtype, partition_info=None):
        fan_in, fan_out = get_fans(x)
        limit = tf.sqrt(6 / (fan_in + fan_out))
        return tf.random_uniform(x, -limit, limit, dtype, seed, name)
    return _glorot_uniform


""" lecun uniform initializer
    initialize within [-limit, limit]:
        limit = sqrt(3 / fan_in)
    where fan_in is the number of inputs units in the weight
"""
def lecun_uniform(seed=None, name=None):
    def _lecun_uniform(x, dtype, partition_info=None):
        fan_in, _ = get_fans(x)
        limit = tf.sqrt( 3 / fan_in)
        return tf.random_uniform(x, -limit, limit, dtype, seed, name)
    return _lecun_uniform


""" lecun normal initializer
    initialize with parameter:
        mean = 0
        stddev = sqrt(1 / fan_in)
    where fan_in is the number of inputs units in the weight
"""
def lecun_normal(seed=None, name=None):
    def _lecun_normal(x, dtype, partition_info=None):
        fan_in, _ = get_fans(x)
        stddev = tf.sqrt(1 / fan_in)
        return tf.random_normal(x, 0, stddev, dtype, seed, name)
    return _lecun_normal


""" truncated normal initializer
"""
def truncated_normal(mean=0.0, stddev=1.0, seed=None, name=None):
    def _truncated_normal(x, dtype, partition_info=None):
        return tf.truncated_normal(x, mean, stddev, dtype, seed, name)
    return _truncated_normal


""" normal initializer
"""
def random_normal(mean=0.0, stddev=1.0, seed=None, name=None):
    def _random_normal(x, dtype, partition_info=None):
        return tf.random_normal(x, mean, stddev, dtype, seed, name)
    return _random_normal


""" uniform initializer
"""
def random_uniform(minval=0, maxval=None, seed=None, name=None):
    def _random_uniform(x, dtype, partition_info=None):
        return tf.random_uniform(x, minval, maxval, dtype, seed, name)
    return _random_uniform

#""" random gamma initializer
#"""
#def random_gamma(alpha, beta=None, dtype=tf.float32, seed=None, name=None):
#    def _random_gamma(x):
#        return tf.random_gamma(x, alpha, beta, dtype, seed, name)
#
#    return _random_gamma
#


"""
"""
def constant(value=0, name=None):
    def _constant(x, dtype, partition_info=None):
        return tf.constant(value, dtype, x, name)
    return _constant

#
#def orthogonal():
#    return tf.orthogonal_initializer

"""
"""
def zeros(name=None):
    def _zeros(x, dtype, partition_info=None):
        return tf.zeros(x, dtype, name)
    return _zeros

def ones(name=None):
    def _ones(x, dtype, partition_info=None):
        return tf.ones(x, dtype, name)
    return _ones


def get(initializer):
    if initializer is None:
        return zeros()
    if isinstance(initializer, str):
        return eval('{}()'.format(initializer))
    elif callable(initializer):
        return initializer()
    # if given list / tuple / np.ndarray value
    # return as lambda function
    elif isinstance(initializer, (list, tuple)):
        return lambda : np.asarray(initializer)
    elif isinstance(initializer, np.ndarray):
        return lambda : initializer
    else:
        raise ValueError('cannot get activates `{}` with type {}'
                         .format(initializer, type(initializer)))
