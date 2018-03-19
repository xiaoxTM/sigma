from .. import ops
from .. import colors
from . import core

def _merge(fun, inputs, output, typename, return_shape):
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if output != xshape:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, xshape, colors.reset))
    if return_shape:
        x = [x, output]
    return x


@core.layer
def concat(inputs,
           axis=-1,
           return_shape=False,
           reuse=False,
           name='concat',
           scope=None):
    inputs_shape = [ops.core.shape(ip) for ip in inputs]
    fun, output = ops.merge.concat(inputs_shape, axis, reuse, name, scope)
    return _merge(fun, inputs, output, 'concatenate', return_shape)


@core.layer
def add(inputs,
        weights=None,
        return_shape=False,
        reuse=False,
        name='add',
        scope=None):
    input_shape = [ops.core.shape(ip) for ip in inputs]
    fun, output = ops.merge.add(input_shape, weights, reuse, name, scope)
    return _merge(fun, inputs, output, 'add', return_shape)


@core.layer
def mul(inputs,
        return_shape=False,
        reuse=False,
        name='mul',
        scope=None):
    input_shape = [ops.core.shape(ip) for ip in inputs]
    fun, output = ops.merge.mul(input_shape, reuse, name, scope)
    return _merge(fun, inputs, output, 'mul', return_shape)
