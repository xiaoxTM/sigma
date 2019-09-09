import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import colors
from sigma.ops import helper, core, mm, actives, convs

def pairwise_euclidean_distance(inputs):
    batch_size, npoints, channels = core.shape(inputs)
   #   [batch-size, npoints, channels]
   #=> [batch-size, npoints, npoints]
   inner = -2 * core.matmul(inputs, core.transpose(inputs, perm=(0,2,1)))
   square = core.sum(core.square(inputs), axis=-1, keepdims=True)
   transpose = core.transpose(square, perm=(0,2,1))
   return square + inner + transpose


def edge_conv(inputs,
              k,
              channels,
              kshape=[1,1],
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
              collection=None,
              summary='histogram',
              reuse=False,
              name=None,
              scope=None):
    """ inputs: tensorflow with shape [batch-size, npoints, channels]
        k: int for knn (top_k neighbours in terms of euclidean distance)
    """
    shape = core.shape(inputs)
    if len(shape) != 3:
        raise ValueError('`inputs` must be [batch-size, npoints, channels], given {}'
                         .format(shape))
    # adjacent matrix: [batch-size, npoints, npoints]
    adjacent_matrix = pairwise_euclidean_distance(inputs)
    def _knn(adjacent_matrix, k, axis=2, selfloop=False):
        # get top_k
        # selfloop: False, exclude self-disance, i.e., no self loop
        # x [batch-size, npoints, channels]
        if selfloop:
            return core.top_k(-adjacent_matrix, k)[1]
        x = core.top_k(-adjacent_matrix, k+1)[1]
        idx = core.range(1, k+1)
        return core.gather(x, idx, axis=axis)

    # idx: [batch-size, npoints, k]
    nn_idx = _knn(adjacent_matrix, k)

    batch_size, npoints, channels = shape
    # [batch-size]
    idx = core.range(batch_size) * npoints
    # [batch-size, 1, 1]
    idx = core.reshape(idx, [batch_size, 1, 1])
    # [batch-size * npoints, channels]
    flatten = core.reshape(inputs, [batch_size * npoints, channels])
    # [batch-size, npoints, k, channels]
    neighbours = core.gather(flatten, nn_idx + idx)
    # [batch-size, npoints, 1, k]
    central = core.expand_dims(inputs,axis=-2)
    # [batch-size, npoints, k, k]
    central = core.tile(central, [1,1,k,1])
    # [batch-size, npoints, k, 2k]
    x = core.concat([central, neighbours-central], axis=-1)
    xshape = core.shape(x)
    x = convs.conv2d(xshape,
                     channels,
                     kshape,
                     stride,
                     padding,
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
                     scope)[0](x)
    return core.max(x, axis=-2, keepdims=True)


def projection_transform(inputs,
                         dims,
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
                         check_input_shape=True,
                         reuse=False,
                         name=None,
                         scope=None):
    input_shape = helper.norm_input_shape(inputs)
    if check_input_shape:
        helper.check_input_shape(input_shape)
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 3:
        raise ValueError('fully_conv require input shape {}[batch-size,'
                         ' dims, channels]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))
    batch_size, indims, incaps = input_shape
    weight_shape = [1, indims, dims, incaps] # get rid of batch_size axis

    bias_shape = [incaps]
    output_shape = [input_shape[0], dims, incaps]
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             'project',
                                             reuse)
    act = actives.get(act)
    weights = mm.malloc('weights',
                        name,
                        weight_shape,
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
                         bias_shape,
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
        bias = 0
    def _projection_transform(x):
        with ops_scope:
            #    [batch-size, indims, incaps]
            #=>  [batch-size, indims, 1, incaps]
            x = core.expand_dims(x, 2)
            #    [batch-size, indims, 1, incaps]
            #  * [1, indims, dims, incaps]
            #=>  [batch-size, indims, dims, incaps] (*)
            #=>  [batch-size, dims, incaps] (sum)
            x = core.sum(x * weights, axis=1) + bias
            return act(x)
    return _projection_transform(inputs)

""" permutation invarnace transformation operation
"""
# @helpers.typecheck(input_shape=list,
#                    dims=int,
#                    channels=int,
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
def permutation_transform(inputs,
                          channels,
                          dims,
                          mode='max',
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
                          check_input_shape=True,
                          reuse=False,
                          name=None,
                          scope=None):
    input_shape = helper.norm_input_shape(inputs)
    if check_input_shape:
        helper.check_input_shape(input_shape)
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 3:
        raise ValueError('fully_conv require input shape {}[batch-size,'
                         ' dims, channels]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))
    batch_size, indims, _ = input_shape
    weight_shape = [1, indims, dims*channels, 1] # get rid of batch_size axis
    bias_shape = [channels]
    output_shape = [input_shape[0], dims, channels]
    if mode == 'max':
        def _extract(x):
            return core.max(x, axis=2)
    elif mode == 'mean':
        def _extract(x):
            return core.mean(x, axis=2)
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             'permutation_transform',
                                             reuse)
    act = actives.get(act)
    weights = mm.malloc('weights',
                        name,
                        weight_shape,
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
                         bias_shape,
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
        bias = 0
    def _permutation_transform(x):
        with ops_scope:
            #    [batch-size, indims, incaps]
            #=>  [batch-size, indims, 1, incaps]
            x = core.expand_dims(x, 2)
            #    [batch-size, indims, 1, incaps]
            #  * [1, indims, dims*channels, 1]
            #=>  [batch-size, indims, dims*channels, incaps] (*)
            #=>  [batch-size, dims*channels, incaps] (sum)
            #=>  [batch-size, dims*channels] (max/mean)
            #=>  [batch-size, dims, channels] (reshape)
            x = _extract(core.sum(x * weights, axis=1))
            return act(core.reshape(x, output_shape) + bias)
    return _permutation_transform(inputs)


def transform_routing(inputs,
                      channels,
                      dims,
                      iterations=2,
                      weight_initializer='random_uniform',
                      weight_regularizer=None,
                      bias_initializer='zeros',
                      bias_regularizer=None,
                      cpuid=0,
                      act='squash',
                      trainable=True,
                      dtype=core.float32,
                      epsilon=core.epsilon,
                      safe=True,
                      collections=None,
                      summary='histogram',
                      reuse=False,
                      name=None,
                      scope=None):
    batch_size, indims, nroutings, incaps = helper.norm_input_shape(inputs)
    output_shape = [batch_size, dims*nroutings, channels] # for concatenating along dims
    logits_shape = [batch_size, 1, channels, incaps]
    bias_shape = [channels, 1, nroutings]
    weight_shape = [1, indims, dims*channels, 1, nroutings]
    weights = mm.malloc('weights',
                        helper.normalize_name(name),
                        weight_shape,
                        dtype,
                        weight_initializer,
                        weight_regularizer,
                        cpuid,
                        trainable,
                        collections,
                        summary,
                        reuse,
                        scope)
    bias = mm.malloc('bias',
                     helper.normalize_name(name),
                     bias_shape,
                     dtype,
                     bias_initializer,
                     bias_regularizer,
                     cpuid,
                     trainable,
                     collections,
                     summary,
                     reuse,
                     scope)
    def _update(idx, x, preds):
        # weight  : [1, indims, dims*channels, 1], shared along incaps
        weight = core.gather(weights, idx, axis=-1)
        #=> subias: [dims, channels, 1], shared along incaps
        subias = core.gather(bias, idx, axis=-1)
        #      x  : [batch-size, dims, nroutings, incaps]
        #=> subx  : [batch-size, indims, incaps]
        subx = core.gather(x, idx, axis=-2)
        #=> subx  : [batch-size, indims, 1, incaps]
        subx = core.expand_dims(subx, 2)
        #=> subx  : [batch-size, indims, dims*channels, incaps] (*)
        #=> subx  : [batch-size, dims*channels, incaps] (sum)
        subx = core.sum(subx * weight, axis=1)
        subx = core.reshape(subx, [batch_size, dims, channels, incaps])
        #=> subx  : [batch-size, dims, channels]
        subx = capsules.dynamic_routing(subx,
                                        logits_shape,
                                        iterations,
                                        subias,
                                        act,
                                        False,
                                        epsilon,
                                        safe)
        preds = preds.write(idx, subx)
        return (idx+1, x, preds)
    def _transform_routing(x):
        predictions = core.TensorArray(dtype=core.float32,
                                       size=nroutings,
                                       clear_after_read=False)
        idx = core.constant(0, core.int32)
        _, _, predictions = core.while_loop(
                lambda idx, x, predictions: idx < iterations,
                _update,
                loop_vars=[idx, x, predictions],
                parallel_iterations=iterations)
        # [nroutings, batch-size, dims, channels]
        predictions = predictions.stack()
        # [batch-size, dims*nroutings, channels]
        predictions = core.transpose(predictions, (1, 2 ,0, 3))
        predictions = core.reshape(predictions, (batch_size, dims*nroutings, channels))
        return predictions
    return _transform_routing(inputs)
