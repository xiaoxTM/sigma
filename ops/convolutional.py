"""
    sigma, a deep neural network framework.
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from .. import colors, helpers
from . import mm
from . import actives
from . import helper
from . import core
import numpy as np
import logging

""" There are two main scopes for sigma to organize tensorflow graph
    - vairable scope : in the form of
    [scope/]layername/variables/{trainable | non-trainable}/variablename
    - operation scope: in the form of [scope/]layername/layertype
    NOTE that separating variable scope from operation scope is to avoid
    tensorflow automatically dispatch different scope-name to operation
    scope even variable-scope and operation-scope are assigned to the
    same string manually.
"""

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
         cpuid=0,
         act=None,
         trainable=True,
         dtype=core.float32,
         bias_axis=-1,
         collections=None,
         summary='histogram',
         reuse=False,
         name=None,
         scope=None):
    # NOTE that the parameters:
    #          `bias_axis`
    #      is setup to deal with de-convolutional op
    #      whose output is kernel_shape[-2]
    #      instead of kernel_shape[-1]
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             # remove `_` from _convop
                                             convop.__name__[1:],
                                             reuse)
    act = actives.get(act)
    weight = mm.malloc('weight',
                       name,
                       kernel_shape,
                       dtype,
                       weight_initializer,
                       weight_regularizer,
                       cpuid,
                       trainable,
                       collections,
                       summary,
                       reuse,
                       scope)
    if not isinstance(bias_initializer, bool) or bias_initializer is True:
        bias = mm.malloc('bias',
                         name,
                         (kernel_shape[bias_axis],),
                         dtype,
                         bias_initializer,
                         bias_regularizer,
                         cpuid,
                         trainable,
                         collections,
                         summary,
                         reuse,
                         scope)
    else:
        bias = False
    def _conv(x):
        with ops_scope:
            x = convop(x, weight)
            if bias:
                x = core.bias_add(x, bias, core.data_format)
            x = act(x)
            return x
    return _conv


""" fully connected operation
"""
# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    padding=str,
#                    trainable=bool,
#                    iterations=int,
#                    collections=str,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def fully_connected(input_shape, nouts,
                    weight_initializer='glorot_uniform',
                    weight_regularizer=None,
                    bias_initializer='zeros',
                    bias_regularizer=None,
                    cpuid=0,
                    act=None,
                    trainable=True,
                    dtype=core.float32,
                    collections=None,
                    summary='histogram',
                    reuse=False,
                    name=None,
                    scope=None):
    helper.check_input_shape(input_shape)
    batch_size = input_shape[0]
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 2:
        raise ValueError('fully_conv require input shape {}[batch-size,'
                         'channels]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))
    kernel_shape = [input_shape[1], nouts] # get rid of batch_size axis
    output_shape = [input_shape[0], nouts]
    def _fully_connected(x, weight):
        return core.matmul(x, weight)
    return conv(_fully_connected,
                kernel_shape,
                weight_initializer,
                weight_regularizer,
                bias_initializer,
                bias_regularizer,
                cpuid,
                act,
                trainable,
                dtype,
                -1,
                collections,
                summary,
                reuse,
                name,
                scope), output_shape


""" 1-D convolutional operation
"""
# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    padding=str,
#                    iterations=int,
#                    collections=str,
#                    trainable=bool,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def conv1d(input_shape, nouts, kshape,
           stride=1,
           padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           cpuid=0,
           act=None,
           trainable=True,
           dtype=core.float32,
           collections=None,
           summary='histogram',
           reuse=False,
           name=None,
           scope=None):
    helper.check_input_shape(input_shape)
    batch_size = input_shape[0]
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 3:
        raise ValueError('conv1d require input shape '
                         '{}[batch-size, cols, channels]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))
    kshape = helper.norm_input_1d(kshape)
    stride = helper.norm_input_id(stride)
    # helper.get_output_shape requires all inputs (except padding)
    # to have same length and be all list / tuple
    output_shape = helper.get_output_shape(input_shape, nouts,
                                           kshape, stride, padding)
    kernel_shape = [kshape[1], input_shape[-1], nouts]
    # conv1d requires stride to be integer
    # collapse dimension into scalar
    stride = stride[1]
    def _conv1d(x, weight):
        return core.conv1d(x, weight, stride, padding.upper())
    return conv(_conv1d,
                kernel_shape,
                weight_initializer,
                weight_regularizer,
                bias_initializer,
                bias_regularizer,
                cpuid,
                act,
                trainable,
                dtype,
                -1,
                collections,
                summary,
                reuse,
                name,
                scope), output_shape


""" 2-D convolutional operation
"""
# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    padding=str,
#                    trainable=bool,
#                    iterations=int,
#                    collections=str,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def conv2d(input_shape, nouts, kshape,
           stride=1,
           padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           cpuid=0,
           act=None,
           trainable=True,
           dtype=core.float32,
           collections=None,
           summary='histogram',
           reuse=False,
           name=None,
           scope=None):
    helper.check_input_shape(input_shape)
    batch_size = input_shape[0]
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 4:
        raise ValueError('conv2d require input shape {}[batch-size, rows,'
                         'cols, channels]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))
    kshape = helper.norm_input_2d(kshape)
    stride = helper.norm_input_2d(stride)
    output_shape = helper.get_output_shape(input_shape, nouts,
                                           kshape, stride, padding)
    kernel_shape = kshape[1:-1] + [input_shape[core.axis], nouts]
    def _conv2d(x, weight):
        return core.conv2d(x, weight, stride, padding.upper())
    return conv(_conv2d,
                kernel_shape,
                weight_initializer,
                weight_regularizer,
                bias_initializer,
                bias_regularizer,
                cpuid,
                act,
                trainable,
                dtype,
                -1,
                collections,
                summary,
                reuse,
                name,
                scope), output_shape


""" 3-D convolutional operation
"""
# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    padding=str,
#                    iterations=int,
#                    collections=str,
#                    trainable=bool,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def conv3d(input_shape, nouts,
           kshape=3,
           stride=1,
           padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           cpuid=0,
           act=None,
           trainable=True,
           dtype=core.float32,
           collections=None,
           summary='histogram',
           reuse=False,
           name=None,
           scope=None):
    helper.check_input_shape(input_shape)
    batch_size = input_shape[0]
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 4:
        raise ValueError('{}conv2d require input shape [batch-size, rows,'
                         'cols, channels], given {}{}'
                         .format(colors.fg.red, input_shape, colors.reset))
    kshape = helper.norm_input_3d(kshape)
    stride = helper.norm_input_3d(stride)
    output_shape = helper.get_output_shape(input_shape, nouts,
                                           kshape, stride, padding)
    kernel_shape = kshape[1:-1] +[input_shape[-1], nouts]
    def _conv3d(x, weight):
        return core.conv3d(x, weight, stride, padding)
    return conv(_conv3d,
                kernel_shape,
                weight_initializer,
                weight_regularizer,
                bias_initializer,
                bias_regularizer,
                cpuid,
                act,
                trainable,
                dtype,
                -1,
                collections,
                summary,
                reuse,
                name,
                scope), output_shape


""" 2-D transpose convolutional operation
"""
# @helpers.typecheck(input_shape=list,
#                    output_shape=list,
#                    nouts=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    padding=str,
#                    iterations=int,
#                    collections=str,
#                    trainable=bool,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def deconv2d(input_shape, output_shape, nout,
             kshape=3,
             stride=1,
             padding='valid',
             weight_initializer='glorot_uniform',
             weight_regularizer=None,
             bias_initializer='zeros',
             bias_regularizer=None,
             cpuid=0,
             act=None,
             trainable=True,
             dtype=core.float32,
             collections=None,
             summary='histogram',
             reuse=False,
             name=None,
             scope=None):
    # NOTE: unlike normal convolutional, whose weights has shape of
    #       => [height, width, inputs_channels, output_channels]
    #       de-convolutional weights has shape of:
    #       => [height, width, output_channels, inputs_channels]
    #       the run-time input shape:
    #       if padding == 'valid'
    #           [batch-size, ceil((output_shape[1:-1] - kshape[1:-1] + 1) / stride[1:-1])]
    #       else:
    #           [batch-size, ceil((output_shape[1:-1]) / stride[1:-1])]
    helper.check_input_shape(input_shape)
    batch_size = input_shape[0]
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 4:
        raise ValueError('{}deconv2d require input shape [batch-size,'
                         'rows, cols, channels], given {}{}'
                         .format(colors.fg.red, input_shape, colors.reset))
    kshape = helper.norm_input_2d(kshape)
    stride = helper.norm_input_2d(stride)
    kernel_shape = kshape[1:-1] +[nout, input_shape[-1]]
    out_shape = output_shape
    if out_shape is None:
        out_shape = input_shape
        out_shape[0] = batch_size
        out_shape[1] *= stride[1]
        out_shape[2] *= stride[2]
        out_shape[3] = nout
    elif isinstance(out_shape, (list, tuple)):
        if len(out_shape) == 2:
            out_shape = [batch_size] + out_shape + [nout]
        elif len(out_shape) == 4 and \
             (out_shape[0] != batch_size or out_shape[-1] != nout):
            raise ValueError('output shape{} not match'
                             ' input_shape{} or hidden units{}'
                             .format(output_shape, input_shape, nout))
    else:
        raise TypeError('out_shape with type `{}` not support'
                        .format(type(out_shape)))
    def _deconv2d(x, weight):
        return core.deconv2d(x, weight, out_shape, stride,
                             padding.upper(), name=name)
    return conv(_deconv2d,
                kernel_shape,
                weight_initializer,
                weight_regularizer,
                bias_initializer,
                bias_regularizer,
                cpuid,
                act,
                trainable,
                dtype,
                -2,
                collections,
                summary,
                reuse,
                name,
                scope), out_shape


# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    mode=str,
#                    padding=str,
#                    iterations=int,
#                    collections=str,
#                    trainable=bool,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def soft_conv(input_shape,
              kshape=3,
              stride=1,
              padding='valid',
              mode='naive',
              weight_initializer='glorot_uniform',
              weight_regularizer=None,
              bias_initializer='zeros',
              bias_regularizer=None,
              offset_weight_initializer='zeros',
              offset_weight_regularizer=None,
              offset_bias_initializer=None,
              offset_bias_regularizer=None,
              cpuid=0,
              act=None,
              trainable=True,
              dtype=core.float32,
              collections=None,
              summary='histogram',
              reuse=False,
              name=None,
              scope=None):
    # //NOTE : soft_conv may not support dynamic batch-size
    # batch_size = input_shape[0]
    # if helper.is_tensor(input_shape):
    #     input_shape = input_shape.as_list()
    helper.check_input_shape(input_shape)
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             'soft_conv',
                                             reuse)
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
    nkernels = np.prod(kshape[:dimlen])
    ndims = np.prod([int(sh / st) for sh, st in zip(dim_shape, dim_stride)])
    # offsets_shape = [-1] + dim_shape + [nkernels, dimlen]
    weight = mm.malloc('weight',
                       name,
                       [nkernels]+kshape[dimlen:],
                       dtype,
                       weight_initializer,
                       weight_regularizer,
                       cpuid,
                       trainable,
                       collections,
                       summary,
                       reuse,
                       scope)
    if not isinstance(bias_initializer, bool) or bias_initializer is True:
        bias = mm.malloc('bias',
                         name,
                         (kshape[-1],),
                         dtype,
                         bias_initializer,
                         bias_regularizer,
                         cpuid,
                         trainable,
                         collections,
                         summary,
                         reuse,
                         scope)
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
        'kshape':kshape,
        'stride':stride,
        'padding':padding,
        'weight_initializer':offset_weight_initializer,
        'weight_regularizer':offset_weight_regularizer,
        'bias_initializer':offset_bias_initializer,
        'bias_regularizer':offset_bias_regularizer,
        'cpuid':cpuid,
        'act':act,
        'trainable':trainable,
        'dtype':dtype,
        'collections':collections,
        'reuse':reuse,
        'summarize':summarize,
        'name':'{}/offsets'.format(name),
        'scope':scope
    }
    #with _scope:
    grids = [np.arange(0, s, t, dtype=np.float32)
              for s,t in zip(dim_shape, dim_stride)]
    grids = np.meshgrid(*grids, indexing='ij')
    grids = np.stack(grids, axis=-1)
    kgrids = [np.arange(-int(k / 2), int(k / 2)+1, dtype=np.float32)
                for k in kshape[:dimlen]]
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
                            [row-idx, col-idx]] with dtype = float32
            Return:        [batch-size,
                            rows * cols * krows * kcols,
                            [batch-idx, row-idx, col-idx]]
        """
        shape = core.tshape(offset_grids)
        batch_range = core.expand_dims(core.range(core.cast(shape[0],
                                                      dtype=core.float32)
                                              ), axis=-1)
        batch_range = core.tile(batch_range, [1, nkernels * ndims])
        int_shape = core.shape(offset_grids)
        batch_range = core.reshape(batch_range, int_shape[:-1] + [1])
        offset_grids = core.concat([batch_range, offset_grids], axis=-1)
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
            offset_grids = core.floor(offset_grids)
        elif mode == 'ceil':
            offset_grids = core.ceil(offset_grids)
        offset_grids = core.cast(offset_grids, dtype=core.int32)
        unstacks = core.unstack(offset_grids, axis=-1)
        for dim, bound in enumerate(dim_shape):
            # dim + 1 to skip batch-idx dimension
            unstacks[dim+dimoffset] = core.clip(unstacks[dim+dimoffset],
                                                       0, bound-1)
        return core.stack(unstacks, axis=-1)

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
        x = core.transpose(x, [axis, 0] + dims)
        if mode in ['naive', 'nearest', 'floor', 'ceil']:
            # [batch-size * rows * cols * krows * kcols, 3]
            # [[batch-idx, row-idx, col-idx], ...]
            offset_grids = _check_bounds_with_cast(offset_grids)
            # [batch-size, rows, cols, channels] =>
            # [channels, batch-size, rows, cols] for map_fn
            gathers = core.map_fn(lambda c:core.gather_nd(c, offset_grids), x)
            # [batch-size, rows * cols * krows * kcols, channels]
        else:
            floor = core.floor(offset_grids)
            ceil  = core.ceil(offset_grids)
            dfloor = 1 - (offset_grids - floor)
            dceil  = 1 - (ceil - offset_grids)

            floor = _check_bounds_with_cast(floor)
            ceil  = _check_bounds_with_cast(ceil)

            # batch-idx: [batch-size, rows * cols * krows * kcols]
            # rows-idx : [batch-size, rows * cols * krows * kcols]
            # cols-idx : [batch-size, rows * cols * krows * kcols]
            # [[batch-idx, rows-idx, col-idx], ...]
            unstack_floor = core.unstack(floor, axis=-1)
            unstack_ceil  = core.unstack(ceil, axis=-1)
            # [[[batch-idx, rows-idx, col-idx], ...]
            #  [[batch-idx, rows-idx, col-idx], ...]]
            combines = [unstack_floor, unstack_ceil]
            unstack_dfloor = core.unstack(dfloor, axis=-1)
            unstack_dceil  = core.unstack(dceil, axis=-1)
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
                    pos = core.stack(pos, axis=-1)
                    # [batch-size, rows * cols * krows * kcols]
                    gather = core.gather_nd(feature_map, pos)
                    gather = gather * factor
                    interpolated.append(core.reshape(gather, shape))
                # [batch-size, rows, cols, channels]
                return core.add(interpolated)

            gathers = core.map_fn(_bilinear, x)
        # [channels, batch-size, rows , cols, krows * kcols]
        shape = [input_shape[axis], -1] + dim_strided_shape + [nkernels]
        gathers = core.reshape(gathers, shape)
        return gathers, offset_grids

    scope = core.name_scope(scope)

    def _soft_conv(x):
        with ops_scope:
            with core.name_scope('offsets'):
                # [batch-size, rows, cols, 2 * krows * kcols]
                offsets = convop(x)
                # [batch-size, rows * cols * krows * kcols, 2]
                offsets = core.reshape(offsets, [-1]+list(grids.shape))
                offset_grids = grids + offsets
            with core.name_scope('gathers'):
                offset_grids = _append_batch_grids(offset_grids)
                # gathers:
                #    [channels, batch-size, rows, cols, krows * kcols]
                gathers, offset_grids = _gather(x, offset_grids)
            with core.name_scope('convolves'):
                x = core.tensordot(gathers, weight, axes=[[0, -1], [1, 0]])
                if bias:
                    x = core.bias_add(x, bias)
            return act(x), offset_grids
    return _soft_conv


# """ 1-D convolutional operation
# """
# def soft_conv1d(input_shape, nouts, kshape, stride,
#                 padding='valid',
#                 weight_initializer=None,
#                 weight_regularizer=None,
#                 bias_initializer=None,
#                 bias_regularizer=None,
#                 act=None, trainable=True,
#                 dtype=core.float32, collections=None,
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
#     kshape = helper.norm_input_1d(kshape)
#     stride = helper.norm_input_id(stride)
#
#     # helper.get_output_shape requires all inputs (except padding)
#     # to have same length and be all list / tuple
#     output_shape = helper.get_output_shape(input_shape, nouts,
#                                            kshape, stride, padding)
#     kernel_shape = [kshape[1], input_shape[-1], nouts]
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
# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    padding=str,
#                    mode=str,
#                    iterations=int,
#                    collections=str,
#                    trainable=bool,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def soft_conv2d(input_shape, nouts,
                kshape=3,
                stride=1,
                padding='valid',
                mode='bilinear',
                weight_initializer='glorot_uniform',
                weight_regularizer=None,
                bias_initializer='zeros',
                bias_regularizer=None,
                offset_weight_initializer='zeros',
                offset_weight_regularizer=None,
                offset_bias_initializer=None,
                offset_bias_regularizer=None,
                cpuid=0,
                act=None,
                trainable=True,
                dtype=core.float32,
                collections=None,
                summary='histogram',
                reuse=False,
                name=None,
                scope=None):
    helper.check_input_shape(input_shape)
    batch_size = input_shape[0]
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 4:
        raise ValueError('conv2d require input shape {}[batch-size, rows,'
                         'cols, channels]{}, given {}'
                        .format(colors.fg.green, colors.reset,
                                colors.red(input_shape)))

    kshape = helper.norm_input_2d(kshape)
    stride = helper.norm_input_2d(stride)
    output_shape = helper.get_output_shape(input_shape, nouts,
                                           kshape, stride, padding)
    kernel_shape = kshape[1:-1] + [input_shape[-1], nouts]
    return soft_conv(input_shape,
                     kernel_shape,
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
                     cpuid,
                     act,
                     trainable,
                     dtype,
                     collections,
                     summary,
                     reuse,
                     name,
                     scope), output_shape


# """ 3-D convolutional operation
# """
# def soft_conv3d(input_shape, nouts, kshape, stride, padding='valid',
#                 weight_initializer=None, weight_regularizer=None,
#                 bias_initializer=None, bias_regularizer=None, act=None,
#                 trainable=True, dtype=core.float32, collections=None,
#                 reuse=False, summarize=True, name=None, scope=None):
#     if name is None:
#         name = helper.dispatch_name('conv3d')
#     if core.is_tensor(inputs):
#         input_shape = inputs.get_shape().as_list()
#     if len(input_shape) != 4:
#         raise ValueError('{}conv2d require input shape [batch-size, rows,'
#                          'cols, channels], given {}{}'
#                          .format(colors.fg.red, input_shape, colors.reset))
#
#     kshape = helper.norm_input_3d(kshape)
#     stride = helper.norm_input_3d(stride)
#
#     output_shape = helper.get_output_shape(input_shape, nouts,
#                                            kshape, stride, padding)
#     kernel_shape = [*kshape[1:-1], input_shape[-1], nouts]
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
#                   nout, kshape, stride,
#                   padding='valid',
#                   weight_initializer=None,
#                   weight_regularizer=None,
#                   bias_initializer=None,
#                   bias_regularizer=None,
#                   act=None, offsets=None,
#                   trainable=True, dtype=core.float32,
#                   collections=None, reuse=False,
#                   summarize=True, name=None, scope=None):
#     # NOTE: unlike normal convolutional, de-convolutional weights has shape of:
#     #       [height, width, output_channels, inputs_channels]
#     #       the run-time input shape:
#     #       if padding == 'valid'
#     #           [batch-size, ceil((output_shape[1:-1] - kshape[1:-1] + 1) / stride[1:-1])]
#     #       else:
#     #           [batch-size, ceil((output_shape[1:-1]) / stride[1:-1])]
#     if name is None:
#         name = helper.dispatch_name('deconv2d')
#     if len(input_shape) != 4:
#         raise ValueError('{}deconv2d require input shape [batch-size,'
#                          'rows, cols, channels], given {}{}'
#                          .format(colors.fg.red, input_shape, colors.reset))
#
#     kshape = helper.norm_input_2d(kshape)
#     stride = helper.norm_input_2d(stride)
#     kernel_shape = [*kshape[1:-1], nout, input_shape[-1]]
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


def sepconv(sepconvop,
            depthwise_kernel,
            pointwise_kernel,
            channel_multiplier,
            weight_initializer='glorot_uniform',
            weight_regularizer=None,
            bias_initializer='zeros',
            bias_regularizer=None,
            cpuid=0,
            act=None,
            trainable=True,
            dtype=core.float32,
            caxis=-1,
            collections=None,
            summary='histogram',
            reuse=False,
            name=None,
            scope=None):
    # NOTE that the parameters:
    #          `caxis`
    #      is setup to deal with de-convolutional op
    #      whose output is kernel_shape[-2]
    #      instead of kernel_shape[-1]
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             sepconvop.__name__[1:],
                                             reuse)
    act = actives.get(act)
    depthwise = mm.malloc('depthwise-weight',
                          name,
                          depthwise_kernel,
                          dtype,
                          weight_initializer,
                          weight_regularizer,
                          cpuid,
                          trainable,
                          collections,
                          summary,
                          reuse,
                          scope)
    pointwise = mm.malloc('pointwise-weight',
                          pointwise_kernel,
                          dtype,
                          weight_initializer,
                          weight_regularizer,
                          cpuid,
                          trainable,
                          collecitons,
                          summary,
                          reuse,
                          name,
                          scope)
    if not isinstance(bias_initializer, bool) or bias_initializer is True:
        bias = mm.malloc('bias',
                         name,
                         (pointwise_kernel[core.axis],),
                         dtype,
                         bias_initializer,
                         bias_regularizer,
                         cpuid,
                         trainable,
                         collections,
                         summary,
                         reuse,
                         scope)
    else:
        bias = False
    def _sepconv(x):
        with ops_scope:
            x = sepconvop(x, depthwise, pointwise)
            if bias:
                x = core.bias_add(x, bias)
            x = act(x)
        return x
    return _sepconv


# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    padding=str,
#                    channel_multiplier=int,
#                    iterations=int,
#                    collections=str,
#                    trainable=bool,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def sepconv2d(input_shape, nouts,
              kshape=3,
              stride=1,
              padding='valid',
              channel_multiplier=1,
              rate=1,
              weight_initializer='glorot_uniform',
              weight_regularizer=None,
              bias_initializer='zeros',
              bias_regularizer=None,
              cpuid=0,
              act=None,
              trainable=True,
              dtype=core.float32,
              collections=None,
              summary='histogram',
              reuse=False,
              name=None,
              scope=None):
    helper.check_input_shape(input_shape)
    batch_size = input_shape[0]
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 4:
        raise ValueError('conv2d require input shape {}[batch-size, rows,'
                         'cols, channels]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))

    kshape = helper.norm_input_2d(kshape)
    stride = helper.norm_input_2d(stride)
    output_shape = helper.get_output_shape(input_shape, nouts,
                                           kshape, stride, padding)
    depthwise_kernel = kshape[1:-1] + [input_shape[core.axis],
                                       channel_multiplier]
    pointwise_kernel = [1, 1,
                        input_shape[core.axis]*channel_multiplier,
                        nouts]
    """
        input: 4-D Tensor with shape according to data_format.
        depthwise_filter: 4-D Tensor with shape
            [filter_height, filter_width, in_channels, channel_multiplier].
            Contains in_channels convolutional filters of depth 1.
        pointwise_filter: 4-D Tensor with shape
            [1, 1, channel_multiplier * in_channels, out_channels].
            Pointwise filter to mix channels after depthwise_filter
            has convolved spatially.
        strides: 1-D of size 4. The strides for the depthwise convolution
            for each dimension of input.
        padding: A string, either 'VALID' or 'SAME'. The padding algorithm.
            See the comment here
        rate: 1-D of size 2. The dilation rate in which we sample input values
            across the height and width dimensions in atrous convolution.
            If it is greater than 1, then all values of strides must be 1.
        name: A name for this operation (optional).
        data_format: The data format for input. Either "NHWC" (default) or "NCHW".
    """
    def _sepconv2d(x, depthwise_filter, pointwise_filter):
        return core.sepconv2d(x, depthwise_filter,
                              pointwise_filter,
                              stride, padding, rate,
                              name, ops.data_format)
    return sepconv(_sepconv2d,
                   depthwise_kernel,
                   pointwise_kernel,
                   channels_multiplier,
                   weight_initializer,
                   weight_regularizer,
                   bias_initializer,
                   bias_regularizer,
                   cpuid,
                   act,
                   trainable,
                   dtype,
                   collections,
                   summary,
                   reuse,
                   name,
                   scope), output_shape
