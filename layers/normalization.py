from ..ops import normalization as norms
from ..ops import helper
import tensorflow as tf
from .core import layer

@layer
def instance_norm(inputs,
                  offset_initializer='zeros',
                  scale_initializer='ones',
                  offset_regularizer=None,
                  scale_regularizer=None,
                  epsilon=0.003,
                  act=None,
                  trainable=True,
                  reuse=False,
                  collections=None,
                  name=None,
                  scope=None):
    input_shape = inputs.get_shape().as_list()
    fun = norms.instance_norm(input_shape=input_shape,
                             offset_initializer=offset_initializer,
                             scale_initializer=scale_initializer,
                             offset_regularizer=offset_regularizer,
                             scale_regularizer=scale_regularizer,
                             epsilon=epsilon, act=act, trainable=trainable,
                             reuse=reuse, collections=collections,
                             name=name, scope=scope)
    x = fun(inputs)
    # helper.print_layer(inputs, x, 'instance_norm', reuse, name)
    if input_shape != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, input_shape, colors.reset,
                                 colors.fg.red, x.get_shape().as_list(),
                                 colors.reset))
    return x


@layer
def batch_norm(inputs,
               momentum=0.99,
               offset_initializer='zeros',
               scale_initializer='ones',
               offset_regularizer=None,
               scale_regularizer=None,
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               epsilon=0.001,
               act=None,
               trainable=True,
               fused=True,
               reuse=False,
               collections=None,
               name=None,
               scope=None):
    input_shape = inputs.get_shape().as_list()
    fun = norms.batch_norm(input_shape=input_shape,
                          momentum=momentum,
                          offset_initializer=offset_initializer,
                          scale_initializer=scale_initializer,
                          offset_regularizer=offset_regularizer,
                          scale_regularizer=scale_regularizer,
                          moving_mean_initializer=moving_mean_initializer,
                          moving_variance_initializer=moving_variance_initializer,
                          epsilon=epsilon, act=act, trainable=trainable,
                          fused=fused, reuse=reuse,
                          collections=collections,
                          name=name, scope=scope)
    x = fun(inputs)
    # helper.print_layer(inputs, x, 'batch_norm', reuse, name)
    if input_shape != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, input_shape, colors.reset,
                                 colors.fg.red, x.get_shape().as_list(),
                                 colors.reset))
    return x


@layer
def dropout(inputs, pkeep, noise_shape=None,
            seed=None, name=None):
    return norms.dropout(pkeep, noise_shape, seed, name)(inputs)
