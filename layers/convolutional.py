import tensorflow as tf
from ..ops import convolutional as convs
from ..ops import helper
from .. import colors

def embedding(inputs, table_size, strategy='mod', dtype=tf.float32,
              initializer=None, regularizer=None, reuse=False,
              trainable=True, collections=None, name=None):
    fun = convs.embedding(table_size, strategy, dtype, initializer,
                          regularizer, trainable, collections, name)
    x = fun(inputs)
    helper.print_layer(inputs, x, 'embedding', reuse, name)
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
def fully_conv(inputs, nouts, weight_initializer='truncated_normal', weight_regularizer=None,
           bias_initializer=None, bias_regularizer=None, act=None, trainable=True, dtype=tf.float32,
           collections=None, reuse=False, summarize=True, name=None, scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = convs.fully_conv(input_shape, nouts, weight_initializer,
                                   weight_regularizer, bias_initializer, bias_regularizer, act,
                                   trainable, dtype, collections, reuse, summarize, name, scope)
    x = fun(inputs)
    helper.print_layer(inputs, x, 'fully_conv', reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape(), colors.reset))
    return x

# alias of fully-conv
dense = fully_conv

""" 1-D convolutional operation
"""
def conv1d(inputs, nouts, kernel, stride, padding='valid', weight_initializer='truncated_normal',
           weight_regularizer=None, bias_initializer=None, bias_regularizer=None, act=None,
           trainable=True, dtype=tf.float32, collections=None, reuse=False, summarize=True, name=None, scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = convs.conv1d(input_shape, nouts, kernel, stride, padding, weight_initializer,
                               weight_regularizer, bias_initializer, bias_regularizer, act,
                               trainable, dtype, collections, reuse, summarize, name, scope)

    x = fun(inputs)
    helper.print_layer(inputs, x, 'conv1d', reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape(), colors.reset))
    return x


""" 2-D convolutional operation
"""
def conv2d(inputs, nouts, kernel, stride, padding='valid', weight_initializer='truncated_normal',
           weight_regularizer=None, bias_initializer=None, bias_regularizer=None, act=None,
           trainable=True, dtype=tf.float32, collections=None, reuse=False, summarize=True,
           name=None, scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = convs.conv2d(input_shape, nouts, kernel, stride, padding, weight_initializer,
                               weight_regularizer, bias_initializer, bias_regularizer, act,
                               trainable, dtype, collections, reuse, summarize, name, scope)
    x = fun(inputs)
    helper.print_layer(inputs, x, 'conv2d', reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape(), colors.reset))
    return x


""" 3-D convolutional operation
"""
def conv3d(inputs, nouts, kernel, stride, padding='valid', weight_initializer='truncated_normal',
          weight_regularizer=None, bias_initializer=None, bias_regularizer=None, act=None,
          trainable=True, dtype=tf.float32, collections=None, reuse=False, summarize=True, name=None, scope=None):
    # TODO: atruous_convxd
    fun, output = convs.conv3d(scope, input_shape, nouts, kernel, stride, padding, weight_initializer,
                               weight_regularizer, bias_initializer, bias_regularizer, act,
                               trainable, dtype, collections, reuse, summarize, name, scope)
    x = fun(inputs)
    helper.print_layer(inputs, x, 'conv3d', reuse, name)
    if output != x.get_shape().as_list():
       raise ValueError('the predicted output shape and the real output shape not match. {}{}{} vs {}{}{}'
                        .format(colors.fg.green, output, colors.reset,
                                colors.fg.red, x.get_shape(), colors.reset))
    return x


""" 2-D transpose convolutional operation
    **TODO** atruos_convxd_tranpose
"""
def deconv2d(inputs, output_shape, nouts, kernel, stride, padding='valid',
             weight_initializer='truncated_normal', weight_regularizer=None, bias_initializer=None,
             bias_regularizer=None, act=None, trainable=True, dtype=tf.float32,
             collections=None, reuse=False, summarize=True, name=None, scope=None):
    input_shape = inputs.get_shape().as_list()
    fun, output = convs.deconv2d(input_shape, output_shape, nouts, kernel, stride, padding,
                                 weight_initializer, weight_regularizer, bias_initializer, bias_regularizer, act,
                                 trainable, dtype, collections, reuse, summarize, name, scope)

    x = fun(inputs)
    helper.print_layer(inputs, x, 'deconv2d', reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real output shape not match. {}{}{} vs {}{}{}'
                         .format(colors.fg.green, output, colors.reset,
                                 colors.fg.red, x.get_shape(), colors.reset))
    return x
