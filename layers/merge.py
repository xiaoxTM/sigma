from ..ops import merge, helper
from .. import colors
import tensorflow as tf

import logging

def concat(inputs, axis=-1, reuse=False, name='concat'):
    inputs_shape = [ip.get_shape().as_list() for ip in inputs]
    fun, output = merge.concat(inputs_shape, axis, name)
    x = fun(inputs)
    helper.print_layer(inputs, x, 'concat', reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape().as_list(), colors.reset))
    return x

def add(inputs, reuse=False, name='add'):
    input_shape = [ip.get_shape().as_list() for ip in inputs]
    fun, output = merge.add(input_shape, name)
    x = fun(inputs)
    helper.print_layer(inputs, x, 'add', reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape().as_list(), colors.reset))
    return x

def mul(inputs, reuse=False, name='mul'):
    input_shape = [ip.get_shape().as_list() for ip in inputs]
    fun, output = merge.mul(input_shape, name)
    x = fun(inputs)
    helper.print_layer(inputs, x, 'mul', reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape().as_list(), colors.reset))
    return x
