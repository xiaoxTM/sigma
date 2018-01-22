import tensorflow as tf
from .. import colors
from ..ops import base, helper
from .core import layer

def placeholder(dtype, shape=None, name=None):
    return tf.placeholder(dtype, shape, name)

@layer
def flatten(inputs, return_shape=False, reuse=False, name=None, scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = base.flatten(input_shape, reuse, name, scope)
    x = fun(inputs)
    # helper.print_layer(inputs, x, 'flatten', reuse, name)
    if output[1:] != x.get_shape().as_list()[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(output, x.get_shape().as_list()))
    if return_shape:
        x = [x, output]
    return x


@layer
def reshape(inputs, output_shape, return_shape=False,
            reuse=False, name=None, scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = base.reshape(output_shape, reuse, name, scope)
    x = fun(inputs)
    # helper.print_layer(inputs, x, 'reshape', reuse, name)
    if output[1:] != output_shape[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(output, output_shape))
    if return_shape:
        x = [x, output]
    return x
