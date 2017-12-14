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


def soft_conv(input_shape, kernel_shape,
              stride=1, padding='valid', mode='naive',
              weight_initializer='glorot_uniform',
              weight_regularizer=None,
              bias_initializer='zeros',
              bias_regularizer=None,
              offset_weight_initializer='zeros',
              offset_weight_regularizer=None,
              offset_bias_initializer=None,
              offset_bias_regularizer=None,
              act=None, trainable=True,
              dtype=tf.float32, axis=-2,
              collections=None, reuse=False,
              summarize=True, name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('soft_conv')
    if scope is None:
        scope = name
    act = actives.get(act)

    input_len = len(input_shape)

    axis = (axis + input_len) % input_len
    dims = list(range(input_len))
    del dims[axis] # remove featrue map dim
    del dims[0]    # remove batch size dim
    dim_shape = [input_shape[dim] for dim in dims]  #[4, 4]
    dimlen = len(dims)
    # [no feature map channel] shape
    ncshape = [-1] + dim_shape # [batchsize, 4, 4]
    nkernels = np.prod(kernel_shape[:dimlen])
    offsets_shape = ncshape + [nkernels, dimlen]

    weight = mm.malloc('{}/weight'.format(name),
                       [nkernels]+kernel_shape[dimlen:], dtype,
                       weight_initializer,
                       weight_regularizer,
                       trainable, collections, reuse, scope)
    # reshaped_weight = tf.reshape(weight, (-1, kernel_shape[-1]))
    if summarize and not reuse:
        tf.summary.histogram(weight.name, weight)
    if not isinstance(bias_initializer, bool) or bias_initializer is True:
        bias = mm.malloc('{}/bias'.format(name),
                         (kernel_shape[-1],), dtype,
                         bias_initializer,
                         bias_regularizer,
                         trainable, collections, reuse, scope)
        if summarize and not reuse:
            tf.summary.histogram(bias.name, bias)
    else:
        bias = False

    if mode not in ['naive', 'nearest', 'bilinear']:
        raise ValueError('`mode` must be one of '
                         '"naive" / "nearest" / "bilinear".'
                         ' given {}'.format(mode))

    kwargs = {
        'input_shape':input_shape,
        'nouts':nkernels * dimlen,
        'kernel':kernel_shape,
        'stride':stride,
        'padding':padding,
        'weight_initializer':offset_weight_initializer,
        'weight_regularizer':offset_weight_regularizer,
        'bias_initializer':offset_bias_initializer,
        'bias_regularizer':offset_bias_regularizer,
        'act':act,
        'trainable':trainable,
        'dtype':dtype,
        'collections':collections,
        'reuse':reuse,
        'summarize':summarize,
        'name':'{}/offsets'.format(name),
        'scope':None
    }
    # print('offset conv parameters:', kwargs)

    #with _scope:
    grids = [np.arange(s, dtype=np.float32) for s in dim_shape]
    grids = np.meshgrid(*grids, indexing='ij')
    grids = np.stack(grids, axis=-1)
    kgrids = [np.arange(-int(k / 2), int(k / 2)+1, dtype=np.float32)
                for k in kernel_shape[:dimlen]]
    kgrids = np.meshgrid(*kgrids, indexing='ij')
    # [[[i, j]_0, [i, j]_1, ... [i, j]_krow]_0,... _kcols]
    kgrids = np.stack(kgrids, axis=-1)
    # [[i, j]_0, [i, j]_1, ..., [i, j]_nkernels]
    kgrids = np.reshape(kgrids, (-1, dimlen))
    # append `nkernels` axis
    grids = np.expand_dims(grids, axis=-2)
    # [rows, cols, 1, 2] => [rows, cols, nkernels, 2]
    grids = np.tile(grids, [1]*(dimlen)+[nkernels]+[1]) + kgrids
    # [[[i_0, j_0, i_1, j_1, ..., i_{nkernels}, j_{nkernels}], ...], ...]
    #    [rows, cols, krowsm * kcols, 2]
    #  =>[rows * cols * krows * kcols, 2]
    grids = grids.reshape((-1, dimlen))

    if input_len == 3: # 1d
        convop = conv1d(**kwargs)[0]
    elif input_len == 4: # 2d
        convop = conv2d(**kwargs)[0]
    elif input_len == 5:
        convop = conv3d(**kwargs)[0]
    else:
        raise ValueError('input shape length must'
                         ' have 3/4/5. given {}'.format(input_len))

    def _append_batch_grids(offset_grids):
        """ offset_grids : [batch-size,
                            rows * cols * krows * kcols,
                            [row-idx, col-idx]] with dtype = int32
            Return:        [batch-size,
                            rows * cols * krows * kcols,
                            [batch-idx, row-idx, col-idx]]
        """
        shape = tf.shape(offset_grids)
        batch_range = tf.expand_dims(tf.range(shape[0]), axis=-1)
        batch_range = tf.tile(batch_range, [1, tf.reduce_prod(shape[1:-1])])
        int_shape = offset_grids.get_shape().as_list()
        batch_range = tf.reshape(batch_range, int_shape[:-1] + [1])
        offset_grids = tf.concat([batch_range, offset_grids], axis=-1)
        return offset_grids

    def _check_bounds_with_cast(offset_grids):
        """ [batch-size,
             rows * cols * krows * kcols,
             [row-idx, col-idx]
            ]
        """
        if mode in ['naive', 'nearest']:
            if mode == 'nearest':
                offset_grids = offset_grids + 0.5
            offset_grids = tf.cast(offset_grids, dtype=tf.int32)
            unstacks = tf.unstack(offset_grids, axis=-1)
            for dim, bound in enumerate(dim_shape):
                unstacks[dim] = tf.clip_by_value(unstacks[dim], 0, bound-1)
            return tf.stack(unstacks, axis=-1)
        else: # bilinear interpolation here
            raise NotImplementedError('bilinear not yet implemented'
                                     ', but coming soon')

    def _gather(x, offset_grids):
        # offset_grids: [batch-size, rows * cols * krows * kcols, [batch-idx, row-idx, col-idx]]
        if mode in ['naive', 'nearest']:
            # [batch-size * rows * cols * krows * kcols, 3]
            # [[batch-idx, row-idx, col-idx], ...]
            gathers = [tf.gather_nd(feature_map, offset_grids)
                          for feature_map in tf.unstack(x, axis=axis)]
            # [batch-size, rows * cols * krows * kcols]
            gathers = tf.stack(gathers, axis=-1)
            # [batch-size, rows , cols, krows * kcols, channels]
            shape = [-1] + dim_shape + [nkernels, input_shape[axis]]
            gathers = tf.reshape(gathers, shape)
        else:
            # [batch-size, rows * cols, krows * kcols, [batch-idx, row-idx, col-idx]]
            offset_grids = tf.reshape(offset_grids, (-1, nkernels, dimlen+1))
        return gathers

    scope = tf.name_scope(scope)

    def _soft_conv(x):
        with scope:
            with tf.name_scope('offsets'):
                # [batch-size, rows, cols, 2 * krows * kcols]
                offsets = convop(x)
                # [batch-size, rows * cols * krows * kcols, 2]
                offsets = tf.reshape(offsets, [-1]+list(grids.shape))
                offset_grids = grids + offsets
            with tf.name_scope('locates'):
                offset_grids = _check_bounds_with_cast(offset_grids)
                offset_grids = _append_batch_grids(offset_grids)
                gathers = _gather(x, offset_grids)
            with tf.name_scope('convolves'):
                x = tf.tensordot(gathers, weight, axes=[[-2, -1], [0, 1]])
                if bias:
                    x = tf.nn.bias_add(x, bias)
            return act(x), offset_grids
    return _soft_conv


# def soft_conv(input_shape, kernel_shape,
#               stride=1, padding='valid', mode='naive',
#               weight_initializer='glorot_uniform',
#               weight_regularizer=None,
#               bias_initializer='zeros',
#               bias_regularizer=None,
#               offset_weight_initializer='zeros',
#               offset_weight_regularizer=None,
#               offset_bias_initializer=None,
#               offset_bias_regularizer=None,
#               act=None, trainable=True,
#               dtype=tf.float32, axis=-1,
#               collections=None, reuse=False,
#               summarize=True, name=None, scope=None):
#     if name is None:
#         name = helper.dispatch_name('soft_conv')
#     if scope is None:
#         scope = name
#     act = actives.get(act)
#
#     # `dim` indicates rows, cols dimension for 2D
#     #                  hidden units for 1D
#     #                  rows, cols, depth for 3D
#     input_len = len(input_shape)
#     axis = (axis + input_len) % input_len
#     dims = list(range(input_len))
#     del dims[axis] # remove featrue map dim
#     del dims[0]    # remove batch size dim
#     dim_shape = [input_shape[dim] for dim in dims]  #[4, 4]
#     dimlen = len(dims)
#     nkernels = np.prod(kernel_shape[:dimlen])
#     ndims = np.prod(dim_shape)
#     offsets_shape =  [-1] + [ndims * nkernels, dimlen]
#     # weight shape: [rows * cols * krows * kcols, nouts]
#     weight = mm.malloc('{}/weight'.format(name),
#                        (nkernels * input_shape[axis], kernel_shape[-1]), dtype,
#                        weight_initializer,
#                        weight_regularizer,
#                        trainable, collections, reuse, scope)
#     if summarize and not reuse:
#         tf.summary.histogram(weight.name, weight)
#     if not isinstance(bias_initializer, bool) or bias_initializer is True:
#         bias = mm.malloc('{}/bias'.format(name),
#                          (kernel_shape[-1],), dtype,
#                          bias_initializer,
#                          bias_regularizer,
#                          trainable, collections, reuse, scope)
#         if summarize and not reuse:
#             tf.summary.histogram(bias.name, bias)
#     else:
#         bias = False
#
#     if mode not in ['naive', 'nearest', 'bilinear']:
#         raise ValueError('`mode` must be one of '
#                          '"naive" / "nearest" / "bilinear".'
#                          ' given {}'.format(mode))
#     out_offset = nkernels * dimlen
#     kwargs = {
#         'input_shape':input_shape,
#         'nouts':out_offset, # change no number of output feature maps
#         'kernel':kernel_shape,
#         'stride':stride,
#         'padding':padding,
#         'weight_initializer':offset_weight_initializer,
#         'weight_regularizer':offset_weight_regularizer,
#         'bias_initializer':offset_bias_initializer,
#         'bias_regularizer':offset_bias_regularizer,
#         'act':act,
#         'trainable':trainable,
#         'dtype':dtype,
#         'collections':collections,
#         'reuse':reuse,
#         'summarize':summarize,
#         'name':'{}/offsets'.format(name),
#         'scope':None
#     }
#
#     # NOTE: we apply grids to dims except *BATCH-SIZE* dim
#     #       since the *BATCH-SIZE* dim generally is None
#     grids = [np.arange(dim) for dim in dim_shape]
#     grids = np.meshgrid(*grids, indexing='ij')
#     indices = [grid.reshape((-1, 1)) for grid in grids]
#     # [[i,j]_0, [i, j]_1, ..., [i, j]_{rows*cols}]
#     # with shape of [rows*cols, 2]
#     indices = np.concatenate(indices, axis=-1)
#
#     kgrids = [np.arange(-int(k / 2), int(k / 2)+1, dtype=np.float32)
#                 for k in kernel_shape[:dimlen]]
#     kgrids = np.meshgrid(*kgrids, indexing='ij')
#     kindices = [np.reshape(kgrid, (-1, 1)) for kgrid in kgrids]
#     # [[i,j]_0, [i, j]_1, ..., [i, j]_{krows*kcols}]
#     # with shape of [krows*kcols, 2]
#     kindices = np.concatenate(kindices, axis=-1)
#
#     # [[i,j]_0, [i, j]_1, ..., [i, j]_{krows*kcols}] ==>
#     # [[i_0,j_0, i_1, j_1, ..., i_{krows*kcols}, j_{krows*kcols}]_0,
#     #   ...,
#     #  [i_0,j_0, i_1, j_1, ..., i_{krows*kcols}, j_{krows*kcols}]_{rows*cols}]
#     # [rows*cols, 2] ==>
#     # [rows*cols, 2*kwors*kcols]
#     indices = np.tile(indices, [1, nkernels])
#     # [rows*cols, kwors*kcols*2]
#     # [rows*cols*kwors*kcols, 2]
#     indices = np.reshape(indices, (-1, dimlen))
#     # [[i,j]_0, [i, j]_1, ..., [i, j]_{krows*kcols}] =>
#     # [[i,j]_0, [i, j]_1, ..., [i, j]_0,
#     #  [i,j]_0, [i, j]_1, ..., [i, j]_1,
#     #  ...
#     #  [i,j]_0, [i, j]_1, ..., [i, j]_{krows*kcols}
#     # ]
#     # [krows*kcols, 2] =>
#     # [rows*cols*kwors*kcols, 2]
#     kindices = np.tile(kindices, [ndims, 1])
#     # add kernel offset
#     # that is, for each pixel in grids (rows x cols)
#     # there are krows x kcols offsets
#     indices = (indices + kindices).astype(np.float32)
#
#     if input_len == 3: # 1d
#         convop = conv1d
#     elif input_len == 4: # 2d
#         convop = conv2d
#     elif input_len == 5:
#         convop = conv3d
#     else:
#         raise ValueError('input shape length must'
#                          ' have 3/4/5. given {}'.format(input_len))
#
#     scope = tf.name_scope(scope)
#
#     def _append_batch_indices(batch_size, indices):
#         """ gathering elements from inputs at coords
#             centered at *one point* o, e.g.,
#             x-----x-----x
#             |     |     |
#             x-----o-----x
#             |     |     |
#             x-----x-----x
#
#             Attributes
#             ==========
#                 inputs : tensor
#                          inputs tensor with the form of
#                          [batch-size, hidden-units, channels] for 1d
#                          [batch-size, rows, cols, channels] for 2d
#                          [batch-size, rows, cols, depths, channels] for 3d
#                 coords : list / tuple of tensors
#                          coordinates for gathering elements
#                          should have form of:
#
#                          for 1d
#                          [[[batch-size * units * channels, 1],  for units batch dim
#                            [batch-size * units * channels, 1],  for units dim
#                            [batch-size * units * channels, 1]], for channels dim
#                            ...
#                          ]
#
#                          for 2d
#                          [[[batch-size * rows * cols * channels, 1],  for batch dim
#                            [batch-size * rows * cols * channels, 1],  for rows dim
#                            [batch-size * rows * cols * channels, 1],  for cols dim
#                            [batch-size * rows * cols * channels, 1]], for channels dim
#                            ...
#                          ]
#
#                          for 3d:
#                          [[[batch-size * rows * cols * depth * channels, 1],  for batch dim
#                            [batch-size * rows * cols * depth * channels, 1],  for rows dim
#                            [batch-size * rows * cols * depth * channels, 1],  for cols dim
#                            [batch-size * rows * cols * depth * channels, 1],  for depth dim
#                            [batch-size * rows * cols * depth * channels, 1]], for channels dim
#                            ...
#                          ]
#         """
#         if mode in ['naive', 'nearest']: # brute cast
#             if mode == 'nearest':
#                 indices = indices + 0.5
#             indices = tf.cast(indices, dtype=tf.int32)
#             batch_range = tf.expand_dims(tf.range(batch_size), axis=-1)
#             batch_range = tf.tile(batch_range, [1, ndims*nkernels])
#             batch_range = tf.reshape(batch_range, [-1, 1])
#             indices = tf.reshape(indices, [-1, dimlen])
#             # print('inside batch coordinates mapping:', indices.get_shape().as_list())
#             indices = tf.concat([batch_range, indices], axis=-1)
#             return indices
#         else: # bilinear
#             positions = []
#             weights = []
#             floor = tf.floor(indices)
#             ceil  = tf.ceil(indices)
#
#             diff_floor = 1 - (indices - floor)
#             diff_ceil  = 1 - (ceil - indices)
#
#             positions.append([floor, ceil])
#             weights.append([diff_floor, diff_ceil])
#             return (positions, weights)
#
#     def _check_bounds(indices):
#         splited = tf.unstack(indices, axis=-1)
#         for idx, dim in enumerate(dim_shape):
#             splited[idx+1] = tf.clip_by_value(splited[idx+1], 0, dim-1)
#         return tf.stack(splited, axis=-1)
#
#     def _gather(x, indices):
#         if mode in ['naive', 'nearest']:
#             # indices : [batch-size * rows * cols * krows * kcols, dimlen]
#             gather = tf.gather_nd(x, indices)
#             return tf.reshape(gather, [-1]+dim_shape+kernel_shape[:dimlen])
#         else:
#             raise NotImplementedError('mode `bilinear` not implemented')
#
#     def _soft_conv(x):
#         with scope:
#             with tf.name_scope('offsets'):
#                 # [batch-size, rows, cols, 2 * krows * kcols]
#                 offsets = convop(**kwargs)[0](x)
#                 # [batch-size, rows, cols, 2, krows * kcols]
#                 offsets = tf.reshape(offsets, offsets_shape)
#                 offsets = tf.concat(offsets, axis=-1)
#             with tf.name_scope('locates'):
#                 # broadcast addition
#                 offset_indices = offsets + indices
#                 offset_indices = _append_batch_indices(tf.shape(x)[0], offset_indices)
#                 # [batch-size, rows, cols, depth] =>
#                 # [[batch-size, rows, cols],
#                 #  [batch-size, rows, cols],
#                 #   ...
#                 #  [batch-size, rows, cols]]
#                 inputs = tf.unstack(x, axis=axis)
#                 offset_indices = _check_bounds(offset_indices)
#                 gathers = [_gather(ip, offset_indices) for ip in inputs]
#                 # [depth, batch-size, rows, cols, krows * kcols]
#                 gathers = tf.stack(gathers, axis=-1)
#                 gathers = tf.reshape(gathers, input_shape[:-1] + [nkernels*input_shape[axis]])
#             with tf.name_scope('convolves'):
#                 # print(gathers.get_shape().as_list())
#                 # print(weight.get_shape().as_list())
#                 x = tf.tensordot(gathers, weight, axes=[[-1], [0]])
#                 if bias:
#                     x = tf.nn.bias_add(x, bias)
#             return act(x), offset_indices
#     return _soft_conv


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
                offset_weight_initializer='zeros',
                offset_weight_regularizer=None,
                offset_bias_initializer=None,
                offset_bias_regularizer=None,
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
    return soft_conv(input_shape,
                     kernel_shape, stride,
                     padding, mode,
                     weight_initializer,
                     weight_regularizer,
                     bias_initializer,
                     bias_regularizer,
                     offset_weight_initializer,
                     offset_weight_regularizer,
                     offset_bias_initializer,
                     offset_bias_regularizer,
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
