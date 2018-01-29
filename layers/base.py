from .. import colors
from ..ops import base, core

@layer
def flatten(inputs,
            return_shape=False,
            reuse=False,
            name=None,
            scope=None):
    input_shape = core.shape(inputs)
    fun, output = base.flatten(input_shape, reuse, name, scope)
    x = fun(inputs)
    xshape = core.shape(x)
    if output[1:] != xshape[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. '
                         '{}{}{} vs {}{}{}'
                         .format(colors.fg.red, output, colors.reset,
                                 colors.fg.green, xshape, colors.reset))
    if return_shape:
        x = [x, output]
    return x


@layer
def reshape(inputs, output_shape, return_shape=False,
            reuse=False, name=None, scope=None):
    input_shape = core.shape(inputs)
    fun, output = base.reshape(output_shape, reuse, name, scope)
    x = fun(inputs)
    if output[1:] != output_shape[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. '
                         '{}{}{} vs {}{}{}'
                         .format(colors.fg.red, output, colors.reset,
                                 colors.fg.green, xshape, colors.reset))
    if return_shape:
        x = [x, output]
    return x
