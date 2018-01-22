from ..ops import merge, helper
from .. import colors
from .core import layer
import tensorflow as tf

def _merge(fun, inputs, output, typename, return_shape):
    x = fun(inputs)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape().as_list(),
                                 colors.reset))
    if return_shape:
        x = [x, output]
    return x


@layer
def concat(inputs,
           axis=-1,
           return_shape=False,
           reuse=False,
           name='concat',
           scope=None):
    inputs_shape = [ip.get_shape().as_list() for ip in inputs]
    fun, output = merge.concat(inputs_shape, axis, reuse, name, scope)
    return _merge(fun, inputs, output, 'concatenate', return_shape)


@layer
def add(inputs,
        return_shape=False,
        reuse=False,
        name='add',
        scope=None):
    input_shape = [ip.get_shape().as_list() for ip in inputs]
    fun, output = merge.add(input_shape, reuse, name, scope)
    return _merge(fun, inputs, output, 'add', return_shape)


@layer
def mul(inputs,
        return_shape=False,
        reuse=False,
        name='mul',
        scope=None):
    input_shape = [ip.get_shape().as_list() for ip in inputs]
    fun, output = merge.mul(input_shape, reuse, name, scope)
    return _merge(fun, inputs, output, 'mul', return_shape)
