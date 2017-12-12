import tensorflow as tf
from .. import colors
from . import mm
from . import actives
from . import helper

import numpy as np

import logging

def embedding(scope, table_shape,
              strategy='mod', dtype=tf.float32,
              initializer='glorot_uniform',
              regularizer=None, trainable=True,
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
def conv(convop, kernel_shape,
         weight_initializer='glorot_uniform',
         weight_regularizer=None,
         bias_initializer='zeros',
         bias_regularizer=None,
         act=None, trainable=True,
         dtype=tf.float32, bias_axis=-1,
         collections=None, reuse=False,
         summarize=True, name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('conv')

    act = actives.get(act)

    # if not reuse:
    #     logging.debug('=====debug===== for convolution:{}'
    #                   .format(convop.__name__))
    #     logging.debug('kernel_shape: {}'.format(kernel_shape))
    #     logging.debug('weight_initializer: {}'.format(weight_initializer))
    #     logging.debug('weight_regularizer: {}'.format(weight_regularizer))
    #     logging.debug('bias_initializer: {}'.format(bias_initializer))
    #     logging.debug('bias_regularizer: {}'.format(bias_regularizer))
    #     logging.debug('act: {}'.format(act.__name__))
    #     logging.debug('trainable: {}'.format(trainable))
    #     logging.debug('dtype: {}'.format(dtype))
    #     logging.debug('bias_axis: {}'.format(bias_axis))
    #     logging.debug('collections: {}'.format(collections))
    #     logging.debug('reuse: {}'.format(reuse))
    #     logging.debug('summarize: {}'.format(summarize))
    #     logging.debug('name: {}'.format(name))
    #     logging.debug('scope: {}'.format(scope))

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
               weight_initializer='glorot_uniform',
               weight_regularizer=None,
               bias_initializer='zeros',
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
def conv1d(input_shape, nouts, kernel,
           stride=1, padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
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
def conv2d(input_shape, nouts, kernel,
           stride=1, padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           act=None, trainable=True,
           dtype=tf.float32, collections=None,
           reuse=False, summarize=True,
           name=None, scope=None):
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
def conv3d(input_shape, nouts, kernel,
           stride=1, padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           act=None, trainable=True,
           dtype=tf.float32, collections=None,
           reuse=False, summarize=True,
           name=None, scope=None):
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
def deconv2d(input_shape, output_shape, nout,
             kernel, stride=1, padding='valid',
             weight_initializer='glorot_uniform',
             weight_regularizer=None,
             bias_initializer='zeros',
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


def soft_conv(input_shape, nouts, kernel_shape,
              stride=1, padding='valid', mode='naive',
              weight_initializer='glorot_uniform',
              weight_regularizer=None,
              bias_initializer='zeros',
              bias_regularizer=None,
              act=None, trainable=True,
              dtype=tf.float32, axis=-2,
              collections=None, reuse=False,
              summarize=True, name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('soft_conv')
    if scope is None:
        scope = name
    act = actives.get(act)

    weight = mm.malloc('{}/weight'.format(name),
                       kernel_shape, dtype,
                       weight_initializer,
                       weight_regularizer,
                       trainable, collections, reuse, scope)
    reshaped_weight = tf.reshape(weight, (-1, *kernel_shape[:-2]))
    if summarize and not reuse:
        tf.summary.histogram(weight.name, weight)
    if not isinstance(bias_initializer, bool) or bias_initializer is True:
        bias = mm.malloc('{}/bias'.format(name),
                         (nouts,), dtype,
                         bias_initializer,
                         bias_regularizer,
                         trainable, collections, reuse, scope)
        if summarize and not reuse:
            tf.summary.histogram(bias.name, bias)
    else:
        bias = False
    input_len = len(input_shape)

    axis = (axis + input_len) % input_len
    dims = list(range(input_len))
    del dims[axis] # remove featrue map dim
    del dims[0]    # remove batch size dim
    dim_len = len(dims)
    positions = [input_shape[dim] for dim in dims]
    kernel_positions = np.prod(kernel_shape[:dim_len])
    input_positions = np.prod(positions)

    if mode not in ['naive', 'nearest', 'bilinear']:
        raise ValueError('`mode` must be one of '
                         '"naive" / "nearest" / "bilinear".'
                         ' given {}'.format(mode))
    out_offset = kernel_positions * dim_len
    kwargs = {
        'input_shape':input_shape,
        'nouts':out_offset, # change no number of output feature maps
        'kernel':kernel_shape,
        'stride':stride,
        'padding':padding,
        'weight_initializer':'zeros',
        'weight_regularizer':weight_regularizer,
        'bias_initializer':bias_initializer,
        'bias_regularizer':bias_regularizer,
        'act':act,
        'trainable':trainable,
        'dtype':dtype,
        'collections':collections,
        'reuse':reuse,
        'summarize':summarize,
        'name':'{}/offsets'.format(name),
        'scope':scope
    }

    # NOTE: we apply grids to dims except *BATCH-SIZE* dim
    #       since the *BATCH-SIZE* dim generally is None
    #       for compiling this code, we set it to 1,
    #       when calculating, we can *REPEAT / TILE* along
    #       *BATCH-SIZE* dim (HOPE THIS WORKS :(
    grids = [tf.range(input_shape[dim]) for dim in dims]
    grids = tf.meshgrid(*grids, indexing='ij')
    indices = [tf.reshape(grid, (-1, 1)) for grid in grids]
    indices = tf.concat(indices, axis=-1)

    kernel_grids = []
    for i in range(dim_len):
        half = int(kernel_shape[i] / 2)
        kernel_grid = tf.range(-half, half+1)
        kernel_grids.append(kernel_grid)
    kernel_grids = tf.meshgrid(*kernel_grids, indexing='ij')
    kernel_indices = [tf.reshape(grid, (-1, 1)) for grid in kernel_grids]
    kernel_indices = tf.concat(kernel_indices, axis=-1)

    # add kernel offset
    indices = tf.tile(indices, [1, kernel_positions])
    indices = tf.reshape(indices, (-1, dim_len))
    kernel_indices = tf.tile(kernel_indices, [input_positions, 1])
    # [rows * cols * krows * kcols, 2]
    indices = tf.cast(indices + kernel_indices, dtype=tf.float32)

    if input_len == 3: # 1d
        convop, offset_shape = conv1d(**kwargs)
    elif input_len == 4: # 2d
        convop, offset_shape = conv2d(**kwargs)
    elif input_len == 5:
        convop, offset_shape = conv3d(**kwargs)
    else:
        raise ValueError('input shape length must'
                         ' have 3/4/5. given {}'.format(input_len))

    _scope = tf.name_scope(scope)

    def _map_batch_coordinates(batch_size, indices):
        """ gathering elements from inputs at coords
            centered at *one point* o, e.g.,
            x-----x-----x
            |     |     |
            x-----o-----x
            |     |     |
            x-----x-----x

            Attributes
            ==========
                inputs : tensor
                         inputs tensor with the form of
                         [batch-size, hidden-units, channels] for 1d
                         [batch-size, rows, cols, channels] for 2d
                         [batch-size, rows, cols, depths, channels] for 3d
                coords : list / tuple of tensors
                         coordinates for gathering elements
                         should have form of:

                         for 1d
                         [[[batch-size * units * channels, 1],  for units batch dim
                           [batch-size * units * channels, 1],  for units dim
                           [batch-size * units * channels, 1]], for channels dim
                           ...
                         ]

                         for 2d
                         [[[batch-size * rows * cols * channels, 1],  for batch dim
                           [batch-size * rows * cols * channels, 1],  for rows dim
                           [batch-size * rows * cols * channels, 1],  for cols dim
                           [batch-size * rows * cols * channels, 1]], for channels dim
                           ...
                         ]

                         for 3d:
                         [[[batch-size * rows * cols * depth * channels, 1],  for batch dim
                           [batch-size * rows * cols * depth * channels, 1],  for rows dim
                           [batch-size * rows * cols * depth * channels, 1],  for cols dim
                           [batch-size * rows * cols * depth * channels, 1],  for depth dim
                           [batch-size * rows * cols * depth * channels, 1]], for channels dim
                           ...
                         ]
        """
        if mode in ['naive', 'nearest']: # brute cast
            if mode == 'nearest':
                indices = indices + 0.5
            indices = tf.cast(indices, dtype=tf.int32)
            batch_range = tf.expand_dims(tf.range(batch_size), axis=-1)
            batch_range = tf.tile(batch_range, [1, input_positions*kernel_positions])
            batch_range = tf.reshape(batch_range, [-1, 1])
            indices = tf.reshape(indices, [-1, dim_len])
            # print('inside batch coordinates mapping:', indices.get_shape().as_list())
            indices = tf.concat([batch_range, indices], axis=-1)
            return indices
        else: # bilinear
            positions = []
            weights = []
            floor = tf.floor(indices)
            ceil  = tf.ceil(indices)

            diff_floor = 1 - (indices - floor)
            diff_ceil  = 1 - (ceil - indices)

            positions.append([floor, ceil])
            weights.append([diff_floor, diff_ceil])
            return (positions, weights)

    def _check_boundry(indices):
        splited = tf.unstack(indices, axis=-1)
        clipped = [tf.clip_by_value(s, 0,
                                    input_shape[dims[idx]]-1)
                   for idx, s in enumerate(splited)]
        return tf.stack(clipped, axis=-1)

    def _gather(x, indices):
        print('x shape:', x.get_shape().as_list())
        print('indices shape:', indices.get_shape().as_list())
        # x : [batch-size, rows, cols]
        # indices:
        #   mode == 'naive' or 'nearest'
        #     i:[[batch-size, rows, cols, krows * kcols],
        #     j: [batch-size, rows, cols, krows * kcols]]
        #   (coords, weights):
        #     coords:
        #       i: [[floor([batch-size, rows, cols, krows * kcols]),
        #             ceil([batch-size, rows, cols, krows * kcols])],
        #       j:  [floor([batch-size, rows, cols, krows * kcols]),
        #             ceil([batch-size, rows, cols, krows * kcols])]]
        # Return:
        #    [rows * cols, batch-size, krows * kcols]
        if mode in ['naive', 'nearest']:
            # [batch-size, rows, cols, krows * kcols] =>
            # [batch-size, rows * cols, krows * kcols]
            gather = tf.gather_nd(x, indices)
            return tf.reshape(gather, [-1]+positions+kernel_shape[:dim_len])
        else:
            raise NotImplementedError('mode `bilinear` not implemented')

    def _soft_conv(x):
        with _scope:
            with tf.name_scope('offsets'):
                # [batch-size, rows, cols, 2 * krows * kcols]
                offsets = convop(x)
                print('learned offsets shape:', offsets.get_shape().as_list())
                # [batch-size, rows, cols, 2, krows * kcols]
                offsets = tf.reshape(offsets,
                        [-1] + offset_shape[1:-1] + [dim_len, kernel_positions])
                offsets = tf.unstack(offsets, axis=-2)
                offsets = [tf.reshape(offset,
                                     (-1, input_positions * kernel_positions, 1)
                                     )
                           for offset in offsets]
                # [[[rows-idx, cols-idx],
                #   [rows-idx, cols-idx],
                #    ... krows * kcols
                #   [rows-idx, cols-idx]]
                #  [[rows-idx, cols-idx],
                #   [rows-idx, cols-idx],
                #    ... krows * kcols
                #   [rows-idx, cols-idx]],
                #
                #   ... batch-size
                #
                #  [[rows-idx, cols-idx],
                #   [rows-idx, cols-idx],
                #    ... krows * kcols
                #   [rows-idx, cols-idx]]
                #  [[rows-idx, cols-idx],
                #   [rows-idx, cols-idx],
                #    ... krows * kcols
                #   [rows-idx, cols-idx]]]
                # [batch-size, rows * cols * krows * kcols, 2]
                offsets = tf.concat(offsets, axis=-1)
                print('adjust offsets shape:', offsets.get_shape().as_list())
            with tf.name_scope('locates'):
                # broadcast addition
                offset_indices = offsets + indices
                # [batch-size, rows * cols * krows * kcols, 2]
                offset_indices = _check_boundry(offset_indices)
                # [batch-size * rows * cols * krows * kcols, 2]
                offset_indices = _map_batch_coordinates(tf.shape(x)[0], offset_indices)
                print('after batch coordinates:', offset_indices.get_shape().as_list())

                # [batch-size, rows, cols, depth] =>
                # [[batch-size, rows, cols],
                #  [batch-size, rows, cols],
                #   ...
                #  [batch-size, rows, cols]]
                inputs = tf.unstack(x, axis=axis)

                gathers = [_gather(ip, offset_indices) for ip in inputs]
                print('gather:', gathers[0].get_shape().as_list())
                # [depth, batch-size, rows, cols, krows * kcols]
                gathers = tf.stack(gathers)
            with tf.name_scope('convolves'):
                # gathers: [depth, batch-size, rows, cols, krows * kcols]
                # reshape weight:                         [krows * kcols, depth, nouts]
                print('gathers shape:', gathers.shape)
                print('weights shape:', weight.shape)
                x = tf.tensordot(gathers, weight, axes=[[0, -2, -1], [0, 1, 2]])
                if bias:
                    x = tf.nn.bias_add(x, bias)
            return act(x), offset_indices
    return _soft_conv


# """ 1-D convolutional operation
# """
# def soft_conv1d(input_shape, nouts, kernel, stride,
#                 padding='valid',
#                 weight_initializer=None,
#                 weight_regularizer=None,
#                 bias_initializer=None,
#                 bias_regularizer=None,
#                 act=None, trainable=True,
#                 dtype=tf.float32, collections=None,
#                 reuse=False, summarize=True,
#                 name=None, scope=None):
#     if name is None:
#         name = helper.dispatch_name('conv1d')
#
#     if len(input_shape) != 3:
#         raise ValueError('conv1d require input shape '
#                          '{}[batch-size, cols, channels]{}, given {}{}{}'
#                          .format(colors.fg.green, colors.reset,
#                                  colors.fg.red, input_shape, colors.reset))
#     kernel = helper.norm_input_1d(kernel)
#     stride = helper.norm_input_id(stride)
#
#     # helper.get_output_shape requires all inputs (except padding)
#     # to have same length and be all list / tuple
#     output_shape = helper.get_output_shape(input_shape, nouts,
#                                            kernel, stride, padding)
#     kernel_shape = [kernel[1], input_shape[-1], nouts]
#
#     return soft_conv(input_shape, nouts, kernel_shape,
#                      weight_initializer, weight_regularizer,
#                      bias_initializer, bias_regularizer,
#                      act, offsets, trainable, dtype, -1,
#                      collections, reuse, summarize,
#                      name, scope), output_shape
#

""" 2-D convolutional operation
"""
def soft_conv2d(input_shape, nouts, kernel, stride=1,
                padding='valid', mode='bilinear',
                weight_initializer='glorot_uniform',
                weight_regularizer=None,
                bias_initializer='zeros',
                bias_regularizer=None,
                act=None, trainable=True,
                dtype=tf.float32, axis=-1,
                collections=None,
                reuse=False, summarize=True,
                name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('conv2d')
    if scope is None:
        scope = name
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

    return soft_conv(input_shape, nouts,
                     kernel_shape, stride,
                     padding, mode,
                     weight_initializer,
                     weight_regularizer,
                     bias_initializer,
                     bias_regularizer,
                     act, trainable, dtype,
                     -1, collections, reuse, summarize,
                     name, scope), output_shape


# """ 3-D convolutional operation
# """
# def soft_conv3d(input_shape, nouts, kernel, stride, padding='valid',
#                 weight_initializer=None, weight_regularizer=None,
#                 bias_initializer=None, bias_regularizer=None, act=None,
#                 trainable=True, dtype=tf.float32, collections=None,
#                 reuse=False, summarize=True, name=None, scope=None):
#     if name is None:
#         name = helper.dispatch_name('conv3d')
#     if tf.is_tensor(inputs):
#         input_shape = inputs.get_shape().as_list()
#     if len(input_shape) != 4:
#         raise ValueError('{}conv2d require input shape [batch-size, rows,'
#                          'cols, channels], given {}{}'
#                          .format(colors.fg.red, input_shape, colors.reset))
#
#     kernel = helper.norm_input_3d(kernel)
#     stride = helper.norm_input_3d(stride)
#
#     output_shape = helper.get_output_shape(input_shape, nouts,
#                                            kernel, stride, padding)
#     kernel_shape = [*kernel[1:-1], input_shape[-1], nouts]
#
#     return soft_conv(input_shape, nouts, kernel_shape,
#                      weight_initializer, weight_regularizer,
#                      bias_initializer, bias_regularizer,
#                      act, offsets, trainable dtype,
#                      -1, collections, reuse, summarize,
#                      name, scope), output_shape
#
#
# """ 2-D transpose convolutional operation
# """
# def soft_deconv2d(input_shape, output_shape,
#                   nout, kernel, stride,
#                   padding='valid',
#                   weight_initializer=None,
#                   weight_regularizer=None,
#                   bias_initializer=None,
#                   bias_regularizer=None,
#                   act=None, offsets=None,
#                   trainable=True, dtype=tf.float32,
#                   collections=None, reuse=False,
#                   summarize=True, name=None, scope=None):
#     # NOTE: unlike normal convolutional, de-convolutional weights has shape of:
#     #       [height, width, output_channels, inputs_channels]
#     #       the run-time input shape:
#     #       if padding == 'valid'
#     #           [batch-size, ceil((output_shape[1:-1] - kernel[1:-1] + 1) / stride[1:-1])]
#     #       else:
#     #           [batch-size, ceil((output_shape[1:-1]) / stride[1:-1])]
#     if name is None:
#         name = helper.dispatch_name('deconv2d')
#     if len(input_shape) != 4:
#         raise ValueError('{}deconv2d require input shape [batch-size,'
#                          'rows, cols, channels], given {}{}'
#                          .format(colors.fg.red, input_shape, colors.reset))
#
#     kernel = helper.norm_input_2d(kernel)
#     stride = helper.norm_input_2d(stride)
#     kernel_shape = [*kernel[1:-1], nout, input_shape[-1]]
#     out_shape = output_shape
#     if out_shape is None:
#         out_shape = input_shape
#         out_shape[1] *= stride[1]
#         out_shape[2] *= stride[2]
#         out_shape[3] = nout
#     elif isinstance(out_shape, (list, tuple)):
#         if len(out_shape) == 2:
#             out_shape = [input_shape[0], *out_shape, nout]
#         elif len(out_shape) == 4 and \
#              (out_shape[0] != input_shape[0] or out_shape[-1] != nout):
#             raise ValueError('output shape not match'
#                              'input_shape and hidden units')
#     else:
#         raise TypeError('out_shape with type `{}` not support'
#                        .format(type(out_shape)))
#
#     return soft_conv(input_shape, nouts, kernel_shape,
#                      weight_initializer, weight_regularizer,
#                      bias_initializer, bias_regularizer,
#                      act, offsets, trainable, dtype,
#                      -2, collections, reuse, summarize,
#                      name, scope), out_shape
