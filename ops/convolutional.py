import tensorflow as tf
from .. import colors
from .. import status
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
    # NOTE that the parameters:
    #          `bias_axis`
    #      is setup to deal with de-convolutional op
    #      whose output is kernel_shape[-2]
    #      instead of kernel_shape[-1]
    if name is None:
        name = helper.dispatch_name('conv')

    act = actives.get(act)

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
    # NOTE: unlike normal convolutional, whose weights has shape of
    #       => [height, width, inputs_channels, output_channels]
    #       de-convolutional weights has shape of:
    #       => [height, width, output_channels, inputs_channels]
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
        raise TypeError('out_shape with type `{}` not support'
                        .format(type(out_shape)))

    def _deconv2d(x, weight):
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
              dtype=tf.float32, collections=None, reuse=False,
              summarize=True, name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('soft_conv')
    if scope is None:
        scope = name
    act = actives.get(act)

    input_len = len(input_shape)

    axis = helper.normalize_axes(input_shape)
    dims = list(range(input_len))
    del dims[axis] # remove featrue map dim
    del dims[0]    # remove batch size dim
    dimlen = len(dims)
    dim_shape = [1] * dimlen
    dim_stride = [1] * dimlen
    dim_strided_shape = [1] * dimlen
    for idx, dim in enumerate(dims):
        dim_shape[idx] = input_shape[dim]
        dim_stride[idx] = stride[dim]
        dim_strided_shape[idx] = int(
                        np.ceil(float(input_shape[dim]) / float(stride[dim])))
    nkernels = np.prod(kernel_shape[:dimlen])
    ndims = np.prod([int(sh / st) for sh, st in zip(dim_shape, dim_stride)])
    # offsets_shape = [-1] + dim_shape + [nkernels, dimlen]

    weight = mm.malloc('{}/weight'.format(name),
                       [nkernels]+kernel_shape[dimlen:], dtype,
                       weight_initializer,
                       weight_regularizer,
                       trainable, collections, reuse, scope)

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

    if mode not in ['naive', 'nearest', 'floor', 'ceil', 'bilinear']:
        raise ValueError('`mode` must be one of '
                         '"naive" / "nearest" / '
                         '"floor" / "ceil" / "bilinear".'
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
    grids = [np.arange(0, s, t, dtype=np.float32)
              for s,t in zip(dim_shape, dim_stride)]
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

    with tf.name_scope('{}/offsets'.format(scope)):
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
                            [row-idx, col-idx]] with dtype = float32
            Return:        [batch-size,
                            rows * cols * krows * kcols,
                            [batch-idx, row-idx, col-idx]]
        """
        shape = tf.shape(offset_grids)
        batch_range = tf.expand_dims(tf.range(tf.cast(shape[0],
                                                      dtype=tf.float32)
                                              ), axis=-1)
        batch_range = tf.tile(batch_range, [1, nkernels * ndims])
        int_shape = offset_grids.get_shape().as_list()
        batch_range = tf.reshape(batch_range, int_shape[:-1] + [1])
        offset_grids = tf.concat([batch_range, offset_grids], axis=-1)
        return offset_grids

    def _check_bounds_with_cast(offset_grids, dimoffset=1):
        """ offset_grids: [batch-size,
                           rows * cols * krows * kcols,
                           [row-idx, col-idx]
                           ] for dimoffset = 0
            offset_grids: [batch-size,
                           rows * cols * krows * kcols,
                           [batch-idx, row-idx, col-idx]
                           ] for dimoffset = 1
            dimoffset : int. dimension offset from the first
                             axis of features
        """
        if mode == 'nearest':
            offset_grids = offset_grids + 0.5
        elif mode == 'floor':
            offset_grids = tf.floor(offset_grids)
        elif mode == 'ceil':
            offset_grids = tf.ceil(offset_grids)

        offset_grids = tf.cast(offset_grids, dtype=tf.int32)
        unstacks = tf.unstack(offset_grids, axis=-1)
        for dim, bound in enumerate(dim_shape):
            # dim + 1 to skip batch-idx dimension
            unstacks[dim+dimoffset] = tf.clip_by_value(unstacks[dim+dimoffset],
                                                       0, bound-1)
        return tf.stack(unstacks, axis=-1)

    def _enumerate(length, nvalue=2):
        """ enumerate all combinations given nvalue and length
            nvalue : int. should be 2
            length : int. the
            for 1d:
                [[0], [1]]
            for 2d:
                [[0, 0], [0, 1], [1, 0], [1, 1]]
            for 3d:
                [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        """
        enum = np.expand_dims(np.arange(nvalue), axis=1)
        for i in range(length-1):
            apps = np.expand_dims(np.arange(nvalue), axis=1)
            apps = np.tile(apps, [1, enum.shape[0]])
            apps = apps.reshape([-1, 1])
            enum = np.tile(enum, [nvalue, 1])
            enum = np.concatenate([apps, enum], axis=1)
        return enum

    def _gather(x, offset_grids):
        # offset_grids: [batch-size, rows * cols * krows * kcols, [batch-idx, row-idx, col-idx]]
        x = tf.transpose(x, [axis, 0] + dims)
        if mode in ['naive', 'nearest', 'floor', 'ceil']:
            # [batch-size * rows * cols * krows * kcols, 3]
            # [[batch-idx, row-idx, col-idx], ...]
            offset_grids = _check_bounds_with_cast(offset_grids)
            # [batch-size, rows, cols, channels] =>
            # [channels, batch-size, rows, cols] for map_fn
            gathers = tf.map_fn(lambda c:tf.gather_nd(c, offset_grids), x)
            # [batch-size, rows * cols * krows * kcols, channels]
        else:
            floor = tf.floor(offset_grids)
            ceil  = tf.ceil(offset_grids)
            dfloor = 1 - (offset_grids - floor)
            dceil  = 1 - (ceil - offset_grids)

            floor = _check_bounds_with_cast(floor)
            ceil  = _check_bounds_with_cast(ceil)

            # batch-idx: [batch-size, rows * cols * krows * kcols]
            # rows-idx : [batch-size, rows * cols * krows * kcols]
            # cols-idx : [batch-size, rows * cols * krows * kcols]
            # [[batch-idx, rows-idx, col-idx], ...]
            unstack_floor = tf.unstack(floor, axis=-1)
            unstack_ceil  = tf.unstack(ceil, axis=-1)
            # [[[batch-idx, rows-idx, col-idx], ...]
            #  [[batch-idx, rows-idx, col-idx], ...]]
            combines = [unstack_floor, unstack_ceil]
            unstack_dfloor = tf.unstack(dfloor, axis=-1)
            unstack_dceil  = tf.unstack(dceil, axis=-1)
            dcombines = [unstack_dfloor, unstack_dceil]

            # combinations::
            #   for 1d:
            #     [[0], [1]]
            #   for 2d:
            #     [[0, 0], [0, 1], [1, 0], [1, 1]]
            #   for 3d:
            #     [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            #      [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
            combinations = _enumerate(dimlen)
            shape = [-1] + dim_strided_shape + [nkernels]

            def _bilinear(feature_map):
                interpolated = []
                # bilinear for each offset point
                for comb in combinations:
                    # initialize with batch-idx dimension
                    pos = [unstack_ceil[0]]
                    factor = 1
                    for idx in range(dimlen):
                        # append [batch-size, rows * cols * krows * kcols]
                        # idx + 1 to skip batch-idx dimension
                        pos.append(combines[comb[idx]][idx+1])
                        factor *= dcombines[comb[idx]][idx+1]

                    # [batch-size, rows * cols * krows * kcols, dimlen]
                    pos = tf.stack(pos, axis=-1)
                    # [batch-size, rows * cols * krows * kcols]
                    gather = tf.gather_nd(feature_map, pos)
                    #gather = tf.stack(gather, axis=-1)
                    gather = gather * factor
                    interpolated.append(tf.reshape(gather, shape))
                # [batch-size, rows, cols, channels]
                return tf.add_n(interpolated)

            gathers = tf.map_fn(_bilinear, x)
        # [channels, batch-size, rows , cols, krows * kcols]
        shape = [input_shape[axis], -1] + dim_strided_shape + [nkernels]
        gathers = tf.reshape(gathers, shape)
        return gathers, offset_grids

    scope = tf.name_scope(scope)

    def _soft_conv(x):
        with scope:
            with tf.name_scope('offsets'):
                # [batch-size, rows, cols, 2 * krows * kcols]
                offsets = convop(x)
                # [batch-size, rows * cols * krows * kcols, 2]
                offsets = tf.reshape(offsets, [-1]+list(grids.shape))
                offset_grids = grids + offsets
            with tf.name_scope('gathers'):
                offset_grids = _append_batch_grids(offset_grids)
                # gathers:
                #    [channels, batch-size, rows, cols, krows * kcols]
                gathers, offset_grids = _gather(x, offset_grids)
            with tf.name_scope('convolves'):
                x = tf.tensordot(gathers, weight, axes=[[0, -1], [1, 0]])
                if bias:
                    x = tf.nn.bias_add(x, bias)
            return act(x), offset_grids
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
                offset_weight_initializer='zeros',
                offset_weight_regularizer=None,
                offset_bias_initializer=None,
                offset_bias_regularizer=None,
                act=None, trainable=True,
                dtype=tf.float32, collections=None,
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
                     collections, reuse, summarize,
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
