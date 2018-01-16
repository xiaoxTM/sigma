from ..ops import merge, helper
from .. import colors
from .layers import layers
import tensorflow as tf

def _merge(fun, inputs, output, typename, return_shape, reuse, name):
    x = fun(inputs)
    # helper.print_layer(inputs, x, typename, reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape().as_list(),
                                 colors.reset))
    if return_shape:
        x = [x, output]
    return x


@layers
def concat(inputs, axis=-1, return_shape=False, reuse=False, name='concat'):
    inputs_shape = [ip.get_shape().as_list() for ip in inputs]
    fun, output = merge.concat(inputs_shape, axis, name)
    return _merge(fun, inputs, output, 'concatenate',
                  return_shape, reuse, name)


@layers
def add(inputs, return_shape=False, reuse=False, name='add'):
    input_shape = [ip.get_shape().as_list() for ip in inputs]
    fun, output = merge.add(input_shape, name)
    return _merge(fun, inputs, output, 'add',
                  return_shape, reuse, name)


@layers
def mul(inputs, return_shape=False, reuse=False, name='mul'):
    input_shape = [ip.get_shape().as_list() for ip in inputs]
    fun, output = merge.mul(input_shape, name)
    return _merge(fun, inputs, output, 'mul',
                  return_shape, reuse, name)
