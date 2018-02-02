from .. import colors
from .. import ops
from . import core

@core.layer
def embedding(inputs, table_size,
              strategy='mod',
              dtype=ops.core.float32,
              initializer='glorot_uniform',
              regularizer=None,
              trainable=True,
              collections=None,
              summarize=True,
              reuse=False,
              name=None,
              scope=None):
    fun = ops.convs.embedding(table_size,
                              strategy,
                              dtype,
                              initializer,
                              regularizer,
                              trainable,
                              collections,
                              summarize,
                              reuse,
                              name,
                              scope)
    x = fun(inputs)
    return x


@core.layer
def flatten(inputs,
            return_shape=False,
            reuse=False,
            name=None,
            scope=None):
    input_shape = ops.core.shape(inputs)
    fun, output = ops.base.flatten(input_shape, reuse, name, scope)
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if output[1:] != xshape[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. '
                         '{}{}{} vs {}{}{}'
                         .format(colors.fg.red, output, colors.reset,
                                 colors.fg.green, xshape, colors.reset))
    if return_shape:
        x = [x, output]
    return x


@core.layer
def reshape(inputs, output_shape,
            return_shape=False,
            reuse=False,
            name=None,
            scope=None):
    input_shape = ops.core.shape(inputs)
    fun, output = ops.base.reshape(output_shape, reuse, name, scope)
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


@core.layer
def input_spec(inputs,
               dtype=ops.core.float32,
               reuse=False,
               name=None,
               scope=None):
    ops_scope, name = ops.helper.assign_scope(name,
                                              scope,
                                              'inputs',
                                              reuse)
    with ops_scope:
        x = ops.core.placeholder(dtype, inputs, name)
        return x
