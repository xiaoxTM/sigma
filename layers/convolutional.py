import tensorflow as tf
from ..ops import convolutional as convs
from ..ops import helper
from .. import colors
from .layers import layers

@layers
def embedding(inputs, table_size,
              strategy='mod',
              dtype=tf.float32,
              initializer='glorot_uniform',
              regularizer=None,
              trainable=True,
              collections=None,
              reuse=False,
              summarize=True,
              name=None,
              scope=None):
    fun = convs.embedding(table_size,
                          strategy,
                          dtype,
                          initializer,
                          regularizer,
                          trainable,
                          collections,
                          reuse,
                          summarize,
                          name,
                          scope)
    x = fun(inputs)
    # helper.print_layer(inputs, x, 'embedding', reuse, name)
    return x


def _layers(fun, inputs, output, return_shape, typename, reuse, name):
    x = fun(inputs)
    # helper.print_layer(inputs, x, typename, reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real'
                         ' output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape(), colors.reset))
    if return_shape:
        x = [x, output]
    return x


""" convolutional layers

    Attribute
    =========
        scope : string
                scope of weights / biases. can be used to retrieve them
        inputs : tensor
                 input tensor
        nouts : int
                number of output channels
        weight_initializer : string | callable function | None
                             initializer to initialize weights when allocate memory
        weight_regularizer : string | None
                             when not None, will be used to calculate with weight
                             penalty for loss to prevent overfitting
        bias_initializer : string | callable function | None
                           initializer to initialize bias when allocate memory
        weight_regularizer : string | None
                             when not None, will be used to calculate with bias
                             penalty for loss to prevent overfitting
        act : string | callable | None
              activation function
        trainable : bool
                    indicates whether weight and bias can be trained or not
                    when back-propagating
        dtype : tensorflow.dtype
                data type of allocated weight / bias
        collections : string | None
                      name of collections. if None, default is GLOBAL
                      (therefore can be initialized by call:
                       ```python
                          session.run(tf.global_variables_initializer())
                       ```
                      )
        reuse : bool
                whether should reuse weight / bias instead of allocat new
        summarize : bool
                    if True, write to tf.summary.histogram
        name : string | None
               variable name when allocating one
"""

""" fully convolutional operation
"""
@layers
def fully_conv(inputs, nouts,
               weight_initializer='glorot_uniform',
               weight_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               act=None,
               trainable=True,
               dtype=tf.float32,
               return_shape=False,
               collections=None,
               reuse=False,
               summarize=True,
               name=None,
               scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = convs.fully_conv(input_shape,
                                   nouts,
                                   weight_initializer,
                                   weight_regularizer,
                                   bias_initializer,
                                   bias_regularizer,
                                   act,
                                   trainable,
                                   dtype,
                                   collections,
                                   reuse,
                                   summarize,
                                   name,
                                   scope)
    return _layers(fun, inputs, output, return_shape,
                   'fully-conv', reuse, name)

# alias of fully-conv
dense = fully_conv

""" 1-D convolutional operation
"""
@layers
def conv1d(inputs, nouts,
           kshape=3,
           stride=1,
           padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           act=None,
           trainable=True,
           dtype=tf.float32,
           return_shape=False,
           collections=None,
           reuse=False,
           summarize=True,
           name=None,
           scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = convs.conv1d(input_shape,
                               nouts,
                               kshape,
                               stride,
                               padding,
                               weight_initializer,
                               weight_regularizer,
                               bias_initializer,
                               bias_regularizer,
                               act,
                               trainable,
                               dtype,
                               collections,
                               reuse,
                               summarize,
                               name,
                               scope)

    return _layers(fun, inputs, output, return_shape,
                   'conv1d', reuse, name)


# @layers
# def soft_conv1d(inputs, nouts, kshape, stride, padding='valid',
#                 offsets=None, offsets_trainable=False,
#                 weight_initializer='glorot_uniform',
#                 weight_regularizer=None,
#                 bias_initializer='zeros',
#                 bias_regularizer=None,
#                 act=None, trainable=True,
#                 dtype=tf.float32,
#                 collections=None,
#                 reuse=False, summarize=True,
#                 name=None, scope=None):
#     input_shape = inputs.get_shape().as_list()
#     fun, output = convs.soft_conv1d(input_shape, nouts,
#                                     kshape, stride, padding,
#                                     weight_initializer,
#                                     weight_regularizer,
#                                     bias_initializer,
#                                     bias_regularizer,
#                                     act, trainable, dtype,
#                                     collections, reuse,
#                                     summarize, name, scope)
#     x = fun(inputs, offsets)
#     return x


""" 2-D convolutional operation
"""
@layers
def conv2d(inputs, nouts,
           kshape=3,
           stride=1,
           padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           act=None,
           trainable=True,
           dtype=tf.float32,
           return_shape=False,
           collections=None,
           reuse=False,
           summarize=True,
           name=None,
           scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = convs.conv2d(input_shape,
                               nouts,
                               kshape,
                               stride, padding,
                               weight_initializer,
                               weight_regularizer,
                               bias_initializer,
                               bias_regularizer,
                               act,
                               trainable,
                               dtype,
                               collections,
                               reuse,
                               summarize,
                               name,
                               scope)
    return _layers(fun, inputs, output, return_shape,
                   'conv2d', reuse, name)


""" 2-D soft convolutional operation
    Returns
    =======
        x if return_offsets is False,
        x, offsets if return_offsets is true.
            mode = 'naive' / 'nearest'
                offsets in the form of [[batch, row, col, channel, kshape],
                                         ...
                                       ] for 2D, e.g., images
            mode = 'bilinear'
                offsets in the form of [[batch, row, col, channel, kernel_topleft],
                                        [batch, row, col, channel, kernel_topright],
                                        [batch, row, col, channel, kernel_bottomleft],
                                        [batch, row, col, channel, kernel_bottomright],
                                         ...
                                       ] for 2D, e.g., images
                where kernel_topleft, kernel_topright,
                      kernel_bottomleft, kernel_bottomright,
                are used to bilinear interpolation
            therefore, 'bilinear' have 4 times of length than 'naive' / 'nearest'
"""
@layers
def soft_conv2d(inputs, nouts,
                kshape=3,
                stride=1,
                padding='valid',
                mode='bilinear',
                return_offsets=False,
                weight_initializer='glorot_uniform',
                weight_regularizer=None,
                bias_initializer='zeros',
                bias_regularizer=None,
                offset_weight_initializer='zeros',
                offset_weight_regularizer=None,
                offset_bias_initializer=None,
                offset_bias_regularizer=None,
                act=None,
                trainable=True,
                dtype=tf.float32,
                return_shape=False,
                collections=None,
                reuse=False,
                summarize=True,
                name=None,
                scope=None):
    # with tf.name_scope(''):
    input_shape = inputs.get_shape().as_list()
    # offset_fun, offset_output = convs.conv2d(input_shape,
    #                                          input_shape[-1]*)
    fun, output = convs.soft_conv2d(input_shape,
                                    nouts,
                                    kshape,
                                    stride,
                                    padding,
                                    mode,
                                    weight_initializer,
                                    weight_regularizer,
                                    bias_initializer,
                                    bias_regularizer,
                                    offset_weight_initializer,
                                    offset_weight_regularizer,
                                    offset_bias_initializer,
                                    offset_bias_regularizer,
                                    act,
                                    trainable,
                                    dtype,
                                    collections,
                                    reuse,
                                    summarize,
                                    name,
                                    scope)
    x, offsets = fun(inputs)
    # helper.print_layer(inputs, x, 'soft_conv2d', reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real'
                         ' output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape(), colors.reset))
    if return_offsets:
        x = [x, offsets]
    if return_shape:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            x = [*x, output]
        else:
            x = [x, output]
    return x


""" 3-D convolutional operation
"""
@layers
def conv3d(inputs, nouts,
           kshape=3,
           stride=1,
           padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           act=None,
           trainable=True,
           dtype=tf.float32,
           return_shape=False,
           collections=None,
           reuse=False,
           summarize=True,
           name=None,
           scope=None):
    # TODO: atruous_convxd
    fun, output = convs.conv3d(scope, input_shape, nouts,
                               kshape, stride, padding,
                               weight_initializer,
                               weight_regularizer,
                               bias_initializer,
                               bias_regularizer, act,
                               trainable, dtype,
                               collections, reuse,
                               summarize, name, scope)
    return _layers(fun, inputs, output, return_shape,
                   'conv3d', reuse, name)


""" 2-D transpose convolutional operation
    **TODO** atruos_convxd_tranpose
"""
@layers
def deconv2d(inputs, output_shape, nouts,
             kshape=3,
             stride=1,
             padding='valid',
             weight_initializer='glorot_uniform',
             weight_regularizer=None,
             bias_initializer='zeros',
             bias_regularizer=None,
             act=None,
             trainable=True,
             dtype=tf.float32,
             return_shape=False,
             collections=None,
             reuse=False,
             summarize=True,
             name=None,
             scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = convs.deconv2d(input_shape, output_shape,
                                 nouts, kshape, stride, padding,
                                 weight_initializer,
                                 weight_regularizer,
                                 bias_initializer,
                                 bias_regularizer,
                                 act, trainable, dtype, collections,
                                 reuse, summarize, name, scope)

    return _layers(fun, inputs, output, return_shape,
                   'deconv2d', reuse, name)


@layers
def sepconv2d(inputs, nouts,
              kshape=3,
              stride=1,
              padding='valid',
              channel_multiplier=1,
              rate=1,
              weight_initializer='glorot_uniform',
              weight_regularizer=None,
              bias_initializer='zeros',
              bias_regularizer=None,
              act=None,
              trainable=True,
              dtype=tf.float32,
              return_shape=False,
              collections=None,
              reuse=False,
              summarize=True,
              name=None,
              scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = convs.sepconv2d(input_shape,
                                  nouts,
                                  kshape,
                                  stride,
                                  padding,
                                  channel_multiplier,
                                  rate,
                                  weight_initializer,
                                  weight_regularizer,
                                  bias_initializer,
                                  bias_regularizer,
                                  act, trainable,
                                  dtype,
                                  collections,
                                  reuse,
                                  summarize,
                                  name,
                                  scope)
    return _layer(fun, inputs, output, return_shape,
                  'sepconv2d', reuse, name)
