from ..ops import normalization as norms
from ..ops import helper, core
from .core import layer

@layer
def instance_norm(inputs,
                  offset_initializer='zeros',
                  scale_initializer='ones',
                  offset_regularizer=None,
                  scale_regularizer=None,
                  epsilon=core.epsilon,
                  act=None,
                  trainable=True,
                  collections=None,
                  reuse=False,
                  name=None,
                  scope=None):
    input_shape = core.shape(inputs)
    fun = norms.instance_norm(input_shape,
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
    xshape = core.shape(x)
    if input_shape != xshape:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, input_shape, colors.reset,
                                 colors.fg.red, xshape, colors.reset))
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
               epsilon=core.epsilon,
               act=None,
               trainable=True,
               fused=True,
               collections=None,
               reuse=False,
               name=None,
               scope=None):
    input_shape = core.shape(inputs)
    fun = norms.batch_norm(input_shape,
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
    xshape = core.shape(x)
    if input_shape != xshape:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, input_shape, colors.reset,
                                 colors.fg.red, xshape, colors.reset))
    return x


@layer
def dropout(inputs, pkeep,
            noise_shape=None,
            seed=None,
            name=None):
    return norms.dropout(pkeep, noise_shape, seed, name)(inputs)
