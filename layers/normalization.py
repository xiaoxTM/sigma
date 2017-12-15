from ..ops import normalization as norm
from ..ops import helper
import tensorflow as tf
from .layers import layers

@layers
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
               fused=True,
               reuse=False,
               collections=None,
               name=None,
               scope=None):
    input_shape = inputs.get_shape().as_list()
    fun = norm.batch_norm(input_shape=input_shape,
                        momentum=momentum,
                        offset_initializer=offset_initializer,
                        scale_initializer=scale_initializer,
                        offset_regularizer=offset_regularizer,
                        scale_regularizer=scale_regularizer,
                        moving_mean_initializer=moving_mean_initializer,
                        moving_variance_initializer=moving_variance_initializer,
                        epsilon=epsilon, act=act,
                        fused=fused, reuse=reuse,
                        collections=collections,
                        name=name, scope=scope)
    x = fun(inputs)
    helper.print_layer(inputs, x, 'add', reuse, name)
    if input_shape != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, input_shape, colors.reset,
                                 colors.fg.red, x.get_shape().as_list(), colors.reset))
    return x

@layers
def dropout(inputs, pkeep, noise_shape=None, seed=None, name=None):
    return norm.dropout(pkeep, noise_shape, seed, name)(inputs)
