import tensorflow as tf
from .. import colors
from . import mm
from . import actives
from . import helper

import logging

def embedding(scope, table_shape, strategy='mod', dtype=tf.float32,
              initializer='random_normal', regularizer=None, trainable=True,
              collections=None, name=None):
    """ table shape should be:
        table-length x embedding-length
        normally, table-length is the number of class
                  emebdding-length

        // TODO: test
    """
    table = mm.malloc(scope, '{}-embedding'.format(name), table_shape, dtype,
                      initializer, regularizer, trainable, collections, name)
    def _embedding(ids):
        return tf.nn.embedding_lookup(table, ids, strategy, name)
    return _embedding


""" base convolutional operation
    Example:
    1. x = convs.conv1d(scope, name, ninput)(input)
       ninput can be input, or int indiciating input channels
    2. conv = convs.conv1d(scope, name, input)
       x = conv(input)
       y = conv(input)
       this will reuse operation parameters (weight, bias)
"""
def conv(convop, kernel_shape, weight_initializer='truncated_normal',
         weight_regularizer=None, bias_initializer=None, bias_regularizer=None,
         act=None, trainable=True, dtype=tf.float32, bias_axis=-1,
         collections=None, reuse=False, summarize=True, name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('conv')

    act = actives.get(act)

    if not reuse:
        logging.debug('=====debug===== for convolution:{}'
                      .format(convop.__name__))
        logging.debug('kernel_shape: {}'.format(kernel_shape))
        logging.debug('weight_initializer: {}'.format(weight_initializer))
        logging.debug('weight_regularizer: {}'.format(weight_regularizer))
        logging.debug('bias_initializer: {}'.format(bias_initializer))
        logging.debug('bias_regularizer: {}'.format(bias_regularizer))
        logging.debug('act: {}'.format(act.__name__))
        logging.debug('trainable: {}'.format(trainable))
        logging.debug('dtype: {}'.format(dtype))
        logging.debug('bias_axis: {}'.format(bias_axis))
        logging.debug('collections: {}'.format(collections))
        logging.debug('reuse: {}'.format(reuse))
        logging.debug('summarize: {}'.format(summarize))
        logging.debug('name: {}'.format(name))
        logging.debug('scope: {}'.format(scope))

    weight = mm.malloc('{}/weight'.format(name), kernel_shape, dtype,
                    weight_initializer, weight_regularizer,
                    trainable, collections, reuse, scope)
    if summarize and not reuse:
        tf.summary.histogram(weight.name, weight)

    if not isinstance(bias_initializer, bool) or bias_initializer is True:
        bias = mm.malloc('{}/bias'.format(name), (kernel_shape[bias_axis],),
                         dtype, bias_initializer, bias_regularizer,
                         trainable, collections, reuse, scope)
        if summarize and not reuse:
            tf.summary.histogram(bias.name, bias)
    else:
        bias = False
    scope = tf.name_scope(name)
    def _conv(x):
        with scope:
            x = convop(x, weight)
            if bias:
                x = tf.nn.bias_add(x, bias)
            x = act(x)
        return x
    return _conv

""" fully convolutional operation
"""
def fully_conv(input_shape, nouts,
               weight_initializer=None,
               weight_regularizer=None,
               bias_initializer=None,
               bias_regularizer=None,
               act=None, trainable=True,
               dtype=tf.float32, collections=None,
               reuse=False, summarize=True,
               name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('fully_conv')

    if len(input_shape) != 2:
        raise ValueError('fully_conv require input shape {}[batch-size,'
                         'channels]{}, given {}{}{}'
                         .format(colors.fg.green, colors.reset,
                                 colors.fg.red, input_shape, colors.reset))
    kernel_shape = [input_shape[1], nouts] # get rid of batch_size axis
    output_shape = [input_shape[0], nouts]
    def _full_conv(x, weight):
        return tf.matmul(x, weight)
    return conv(convop=_full_conv,
                kernel_shape=kernel_shape,
                weight_initializer=weight_initializer,
                weight_regularizer=weight_regularizer,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
                act=act, trainable=trainable, dtype=dtype,
                bias_axis=-1, collections=collections,
                reuse=reuse, summarize=summarize,
                name=name, scope=scope), output_shape

""" 1-D convolutional operation
"""
def conv1d(input_shape, nouts, kernel, stride,
           padding='valid',
           weight_initializer=None,
           weight_regularizer=None,
           bias_initializer=None,
           bias_regularizer=None,
           act=None, trainable=True,
           dtype=tf.float32, collections=None,
           reuse=False, summarize=True,
           name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('conv1d')

    if len(input_shape) != 3:
        raise ValueError('conv1d require input shape '
                         '{}[batch-size, cols, channels]{}, given {}{}{}'
                         .format(colors.fg.green, colors.reset,
                                 colors.fg.red, input_shape, colors.reset))
    kernel = helper.norm_input_1d(kernel)
    stride = helper.norm_input_id(stride)

    # helper.get_output_shape requires all inputs (except padding)
    # to have same length and be all list / tuple
    output_shape = helper.get_output_shape(input_shape, nouts,
                                           kernel, stride, padding)
    kernel_shape = [kernel[1], input_shape[-1], nouts]
    # tf.nn.conv1d requires stride to be integer
    # collapse dimension into scalar
    stride = stride[1]

    def _conv1d(x, weight):
        return tf.nn.conv1d(x, weight, stride, padding.upper())
    return conv(convop=_conv1d, kernel_shape=kernel_shape,
                weight_initializer=weight_initializer,
                weight_regularizer=weight_regularizer,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
                act=act, trainable=trainable, dtype=dtype,
                bias_axis=-1, collections=collections,
                reuse=reuse, summarize=summarize,
                name=name, scope=scope), output_shape

""" 2-D convolutional operation
"""
def conv2d(input_shape, nouts, kernel, stride, padding='valid',
           weight_initializer=None, weight_regularizer=None,
           bias_initializer=None, bias_regularizer=None, act=None,
           trainable=True, dtype=tf.float32, collections=None,
           reuse=False, summarize=True, name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('conv2d')
    if len(input_shape) != 4:
        raise ValueError('conv2d require input shape {}[batch-size, rows,'
                         'cols, channels]{}, given {}{}{}'
                        .format(colors.fg.green, colors.reset, colors.fg.red,
                                input_shape, colors.reset))

    kernel = helper.norm_input_2d(kernel)
    stride = helper.norm_input_2d(stride)
    output_shape = helper.get_output_shape(input_shape, nouts,
                                           kernel, stride, padding)
    kernel_shape = [*kernel[1:-1], input_shape[-1], nouts]
    # print('kernel:', kernel)
    # print('stride:', stride)
    # print('kernel shape:', kernel_shape)
    # print('output_shape:', output_shape)

    def _conv2d(x, weight):
        return tf.nn.conv2d(x, weight, stride, padding.upper())
    return conv(convop=_conv2d, kernel_shape=kernel_shape,
                weight_initializer=weight_initializer,
                weight_regularizer=weight_regularizer,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
                act=act, trainable=trainable, dtype=dtype,
                bias_axis=-1, collections=collections,
                reuse=reuse, summarize=summarize,
                name=name, scope=scope), output_shape


""" 3-D convolutional operation
"""
def conv3d(input_shape, nouts, kernel, stride, padding='valid',
           weight_initializer=None, weight_regularizer=None,
           bias_initializer=None, bias_regularizer=None, act=None,
           trainable=True, dtype=tf.float32, collections=None,
           reuse=False, summarize=True, name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('conv3d')
    if tf.is_tensor(inputs):
        input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('{}conv2d require input shape [batch-size, rows,'
                         'cols, channels], given {}{}'
                         .format(colors.fg.red, input_shape, colors.reset))

    kernel = helper.norm_input_3d(kernel)
    stride = helper.norm_input_3d(stride)

    output_shape = helper.get_output_shape(input_shape, nouts,
                                           kernel, stride, padding)
    kernel_shape = [*kernel[1:-1], input_shape[-1], nouts]

    def _conv3d(x, weight):
        return tf.nn.conv3d(x, weight, stride, padding)
    return conv(convop=_conv3d, kernel_shape=kernel_shape,
                weight_initializer=weight_initializer,
                weight_regularizer=weight_regularizer,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
                act=act, trainable=trainable, dtype=dtype,
                bias_axis=-1, collections=collections,
                reuse=reuse, summarize=summarize,
                name=name, scope=scope), output_shape


""" 2-D transpose convolutional operation
"""
def deconv2d(input_shape, output_shape, nout, kernel, stride,
             padding='valid',
             weight_initializer=None,
             weight_regularizer=None,
             bias_initializer=None,
             bias_regularizer=None,
             act=None, trainable=True, dtype=tf.float32,
             collections=None, reuse=False,
             summarize=True, name=None, scope=None):
    # NOTE: unlike normal convolutional, de-convolutional weights has shape of:
    #       [height, width, output_channels, inputs_channels]
    #       the run-time input shape:
    #       if padding == 'valid'
    #           [batch-size, ceil((output_shape[1:-1] - kernel[1:-1] + 1) / stride[1:-1])]
    #       else:
    #           [batch-size, ceil((output_shape[1:-1]) / stride[1:-1])]
    if name is None:
        name = helper.dispatch_name('deconv2d')
    if len(input_shape) != 4:
        raise ValueError('{}deconv2d require input shape [batch-size,'
                         'rows, cols, channels], given {}{}'
                         .format(colors.fg.red, input_shape, colors.reset))

    kernel = helper.norm_input_2d(kernel)
    stride = helper.norm_input_2d(stride)
    kernel_shape = [*kernel[1:-1], nout, input_shape[-1]]
    out_shape = output_shape
    if out_shape is None:
        out_shape = input_shape
        out_shape[1] *= stride[1]
        out_shape[2] *= stride[2]
        out_shape[3] = nout
    elif isinstance(out_shape, (list, tuple)):
        if len(out_shape) == 2:
            out_shape = [input_shape[0], *out_shape, nout]
        elif len(out_shape) == 4 and \
             (out_shape[0] != input_shape[0] or out_shape[-1] != nout):
            raise ValueError('output shape not match'
                             'input_shape and hidden units')
    else:
        raise TypeError('out_shape with type `{}` not support'.format(type(out_shape)))
    #
    # print('input shape:', input_shape)
    # print('kernel shape:', kernel_shape)
    # print('stride:', stride)
    # print('output shape:', out_shape)

    def _deconv2d(x, weight):
        # out = tf.nn.conv2d_transpose(x, weight, out_shape, stride,
        #                              padding.upper(), name=name)
        # out.set_shape(out_shape)
        # return out
        return tf.nn.conv2d_transpose(x, weight, out_shape, stride,
                                      padding.upper(), name=name)
    return conv(convop=_deconv2d, kernel_shape=kernel_shape,
                weight_initializer=weight_initializer,
                weight_regularizer=weight_regularizer,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
                act=act, trainable=trainable, dtype=dtype,
                bias_axis=-2, collections=collections,
                reuse=reuse, summarize=summarize,
                name=name, scope=scope), out_shape
