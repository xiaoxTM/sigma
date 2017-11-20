import tensorflow as tf
from .. import colors
from ..ops import base, helper
from .layers import layers

@layers
def flatten(inputs, reuse=False, name=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = base.flatten(input_shape, name)
    x = fun(inputs)
    if not reuse:
        helper.print_layer(inputs, x, 'flatten', reuse, name)
    if output[1:] != x.get_shape().as_list()[1:]:
        raise ValueError('the predicted output shape and the real output shape not match. {} vs {}'
                        .format(output, x.get_shape().as_list()))
    return x

@layers
def reshape(inputs, output_shape, reuse=False, name=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = base.reshape(output_shape, name)
    x = fun(inputs)
    if not reuse:
        helper.print_layer(inputs, x, 'reshape', reuse, name)
    if output[1:] != output_shape[1:]:
        raise ValueError('the predicted output shape and the real output shape not match. {} vs {}'
                         .format(output, output_shape))
    return x
