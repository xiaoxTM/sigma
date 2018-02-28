from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .. import ops
from . import core

@core.layer
def instance_norm(inputs,
                  offset_initializer='zeros',
                  scale_initializer='ones',
                  offset_regularizer=None,
                  scale_regularizer=None,
                  epsilon=ops.core.epsilon,
                  act=None,
                  trainable=True,
                  collections=None,
                  reuse=False,
                  name=None,
                  scope=None):
    input_shape = ops.core.shape(inputs)
    fun = ops.norms.instance_norm(input_shape,
                                  offset_initializer,
                                  scale_initializer,
                                  offset_regularizer,
                                  scale_regularizer,
                                  epsilon,
                                  act,
                                  trainable,
                                  collections,
                                  reuse,
                                  name,
                                  scope)
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if input_shape != xshape:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, input_shape, colors.reset,
                                 colors.fg.red, xshape, colors.reset))
    return x


@core.layer
def conditional_instance_norm(inputs,
                              bank_size,
                              offset_initializer='zeros',
                              scale_initializer='ones',
                              offset_regularizer=None,
                              scale_regularizer=None,
                              epsilon=ops.core.epsilon,
                              act=None,
                              trainable=True,
                              collections=None,
                              reuse=False,
                              name=None,
                              scope=None):
    input_shape = ops.core.shape(inputs)
    fun = ops.norms.conditional_instance_norm(input_shape,
                                              bank_size,
                                              offset_initializer,
                                              scale_initializer,
                                              offset_regularizer,
                                              scale_regularizer,
                                              epsilon,
                                              act,
                                              trainable,
                                              collections,
                                              reuse,
                                              name,
                                              scope)
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if input_shape != xshape:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, input_shape, colors.reset,
                                 colors.fg.red, xshape, colors.reset))
    return x


@core.layer
def batch_norm(inputs,
               momentum=0.99,
               offset_initializer='zeros',
               scale_initializer='ones',
               offset_regularizer=None,
               scale_regularizer=None,
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               epsilon=ops.core.epsilon,
               act=None,
               trainable=True,
               fused=True,
               collections=None,
               reuse=False,
               name=None,
               scope=None):
    input_shape = ops.core.shape(inputs)
    fun = ops.norms.batch_norm(input_shape,
                               momentum,
                               offset_initializer,
                               scale_initializer,
                               offset_regularizer,
                               scale_regularizer,
                               moving_mean_initializer,
                               moving_variance_initializer,
                               epsilon,
                               act,
                               trainable,
                               fused,
                               collections,
                               reuse,
                               name,
                               scope)
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if input_shape != xshape:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, input_shape, colors.reset,
                                 colors.fg.red, xshape, colors.reset))
    return x


@core.layer
def dropout(inputs, pkeep,
            noise_shape=None,
            seed=None,
            name=None):
    return ops.norms.dropout(pkeep, noise_shape, seed, name)(inputs)
