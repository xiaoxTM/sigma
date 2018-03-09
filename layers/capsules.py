from .. import ops, colors
from . import core, convs


@core.layer
def fully_connected(inputs, nouts, caps_dims,
                    iterations=2,
                    leaky=False,
                    weight_initializer='glorot_uniform',
                    weight_regularizer=None,
                    logits_initializer='zeros', # no bias
                    logits_regularizer=None,
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
    fun, output = ops.capsules.fully_connected(input_shape,
                                               nouts,
                                               caps_dims,
                                               iterations,
                                               leaky,
                                               weight_initializer,
                                               weight_regularizer,
                                               logits_initializer,
                                               logits_regularizer,
                                               act,
                                               trainable,
                                               dtype,
                                               collections,
                                               summarize,
                                               reuse,
                                               name,
                                               scope)
    return convs._layers(fun, inputs, output, return_shape,
                         'caps_fully_connected', reuse, name)

dense = fully_connected

@core.layer
def conv2d(inputs, nouts, caps_dims, kshape,
           iterations=2,
           leaky=False,
           stride=1,
           padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           logits_initializer='zeros',
           logits_regularizer=None,
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
                                      weight_initializer,
                                      weight_regularizer,
                                      logits_initializer,
                                      logits_regularizer,
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
