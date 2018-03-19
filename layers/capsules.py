from .. import ops, colors
from . import core, convs


@core.layer
def norm(inputs,
         axis=None,
         act=None,
         return_shape=False,
         reuse=False,
         name=None,
         scope=None):
    """ norm of vector
        input vector output scalar
    """
    input_shape = ops.core.shape(inputs)
    fun, output = ops.capsules.norm(input_shape,
                                    axis,
                                    act,
                                    reuse,
                                    name,
                                    scope)
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
def dot(inputs, nouts, caps_dims,
        iterations=2,
        leaky=False,
        weight_initializer='glorot_uniform',
        weight_regularizer=None,
        bias_initializer='zeros', # no bias
        bias_regularizer=None,
        act=None,
        trainable=True,
        dtype=ops.core.float32,
        return_shape=False,
        collections=None,
        summarize=True,
        reuse=False,
        name=None,
        scope=None):
    """ dot operation inside a capsule. That is:
        ----     ---------
        |c1|     |w11|w12|     --------------------------
        |--|     ---------     |c1*w11 + c2*w12 + c3*w13|
        |c2| dot |w12|w22| ==> --------------------------
        |--|     ---------     |c1*w21 + c2*w22 + c3*w23|
        |c3|     |w13|w23|     --------------------------
        ----     ---------
        NOTE in ''Dynamtic Routing Between Capsules``
        this operations is called fully_connected layer
        (a.k.a., dense layer)
    """
    input_shape = ops.core.shape(inputs)
    fun, output = ops.capsules.dot(input_shape,
                                   nouts,
                                   caps_dims,
                                   iterations,
                                   leaky,
                                   weight_initializer,
                                   weight_regularizer,
                                   bias_initializer,
                                   bias_regularizer,
                                   act,
                                   trainable,
                                   dtype,
                                   collections,
                                   summarize,
                                   reuse,
                                   name,
                                   scope)
    return convs._layers(fun, inputs, output, return_shape,
                         'caps_dot', reuse, name)


@core.layer
def conv2d(inputs, nouts, caps_dims, kshape,
           iterations=3,
           leaky=False,
           stride=1,
           padding='valid',
           fastmode=False,
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           act=None,
           trainable=True,
           dtype=ops.core.float32,
           return_shape=False,
           collections=None,
           summarize=True,
           reuse=False,
           name=None,
           scope=None):
    input_shape = ops.core.shape(inputs)
    fun, output = ops.capsules.conv2d(input_shape,
                                      nouts,
                                      caps_dims,
                                      kshape,
                                      iterations,
                                      leaky,
                                      stride,
                                      padding,
                                      fastmode,
                                      weight_initializer,
                                      weight_regularizer,
                                      bias_initializer,
                                      bias_regularizer,
                                      act,
                                      trainable,
                                      dtype,
                                      collections,
                                      summarize,
                                      reuse,
                                      name,
                                      scope)
    return convs._layers(fun, inputs, output, return_shape,
                         'caps_conv2d', reuse, name)
