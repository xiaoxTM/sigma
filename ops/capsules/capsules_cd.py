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

from .. import colors
from . import core, actives, helper, mm
import logging

from .. import helpers

# capsule networks with dynamic routing
# NOTE: tensor shape:
#       1> for fully_connected capsule:
#           [batch-size, caps, dims]
#vs normal: [batch-size, caps]
#       2> for conv1d capsules:
#           [batch-size, neurons,  caps, dims]
#vs normal: [batch-size, neurons, caps]
#       3> for conv2d capsules:
#           [batch-size, rows, cols, caps, dims]
#vs normal: [batch-size, rows, cols, caps]


def _leaky_routing(logits):
    """ leaky routing
        This enables active capsules to be routed to the added
        parent capsule if they are not good fit for any of the
        parent capsules

        Attributes
        ==========
        logits : Tensor
                 output tensor of one layer with shape of:
                 [batch-size, ncaps, caps_dim]
                 for fully connected
                 or
                 [batch-size, rows, cols, ncaps, caps_dim]
                 for conv2d
                 NOTE that, logits is not `activated` by
                 `softmax` or `sigmoid`
    """
    shape = core.shape(logits)
    nouts = shape[core.caxis]
    shape[core.caxis] = 1
    leak = core.zeros(shape)
    # add an extra dimension to routes
    leaky_logits = core.concat([leak, logits], axis=core.caxis)
    # routes to capsules who has max probability
    leaky_routs = core.softmax(leaky_logits, axis=core.caxis)
    # remove added dimension
    return core.split(leaky_routs, [1, nouts], axis=core.caxis)[1]


def dynamic_routing(prediction,
                    logits_shape,
                    iterations,
                    bias,
                    act='squash',
                    leaky=False, # not used in current version
                    epsilon=core.epsilon,
                    safe=True):
    """ calculate v_j by dynamic routing
        Attributes
        ==========
        prediction : Tensor
                     predictions from previous layers
                     denotes u^{\hat}_{j|i} in paper
                     with the shape of
                     [batch-size, caps, dims, incaps]
                     for fully_connected, and
                     [batch-size, neurons, caps, dims, incaps]
                     for conv1d, and
                     [batch-size, rows, cols, caps, dims, incaps]
                     for conv2d
        logits_shape : Tensor
                       denotes b_{i,j} in paper
                       with the shape of
                       [batch-size, caps, 1, incaps]
                       for fully_connected, and
                       [batch-size, 1, caps, 1, incaps]
                       for conv1d, and
                       [batch-size, 1, 1, caps, 1, incaps]
                       for conv2d
        iterations : int
                     iteration times to adjust c_{i, j}.
                     donates r in paper
        leaky : boolean
                whether leaky_routing or not

        Returns
        ==========
        Activated tensor of the output layer. That is v_j
        with shape of
        [batch-size, caps, dims]
        for fully_connected
        [batch-size, neurons, caps, dims]
        for conv1d
        [batch-size, rows, cols, caps, dims]
        for conv2d
    """
    # restore v_j
    activations = core.TensorArray(dtype=core.float32,
                                   size=iterations,
                                   clear_after_read=False)
    logits = core.zeros(logits_shape, dtype=core.float32)
    act = actives.get(act, axis=-2)
    idx = core.constant(0, dtype=core.int32)
    # softmax along with `caps` axis
    no_grad_prediction = core.stop_gradient(prediction)

    def last(idx, activations, logits):
        # softmax along with `caps` axis
        # that is, `agree` on some capsule
        # a.k.a., which capsule in higher layer to activate
        #        logits: [batch-size, caps, 1, incaps]
        #              / [batch-size, neurons, caps, 1, incaps]
        #              / [batch-size, rows, cols, caps, 1, incaps]
        #=>coefficients: [batch-size, caps, 1, incaps]
        #              / [batch-size, neurons, caps, 1, incaps]
        #              / [batch-size, rows, cols, caps, 1, incaps]
        coefficients = core.softmax(logits, axis=-3)
        # bias: [caps, dims, 1]
        # coefficients * prediction:
        #=> [batch-size, caps, dims, incaps] (i.e., elements in the same capsules shares bias)
        # sum operation (i.e., preactivate):
        #=> [batch-size, caps, dims, 1]
        preactive = core.sum(coefficients * prediction,
                             axis=-1,
                             keepdims=True) + bias
        activation = act(preactive)
        activations = activations.write(idx, activation)
        return activations, logits

    def call(idx, activations, logits):
        coefficients = core.softmax(logits, axis=-3)
        preactive = core.sum(coefficients * no_grad_prediction,
                             axis=-1,
                             keepdims=True) + bias
        activation = act(preactive)
        activations = activations.write(idx, activation)
        # sum up along `dims`
        # prediction * activation:
        #=> [batch-size, caps, dims, incaps]
        # * [batch-size, caps, dims, 1]
        #=> [batch-size, caps, dims, incaps]
        # sum:
        #=> [batch-size, caps, 1, incaps]
        distance = core.sum(no_grad_prediction * activation,
                            axis=-2,
                            keepdims=True)
        logits += distance
        return activations, logits

    def _update(i, logits, activations):
        """ dynamic routing to update coefficiences (c_{i, j})
            logits : [batch-size, /*rows, cols,*/ incaps, caps, 1]
        """
        activations, logits = core.cond(core.eq(i+1, iterations),
                lambda: last(i, activations, logits),
                lambda: call(i, activations, logits))
        return (i+1, logits, activations)

    _, logits, activations = core.while_loop(
        lambda idx, logits, activations: idx < iterations,
        _update,
        loop_vars=[idx, logits, activations],
        swap_memory=True)
    #   [batch-size, nrows, ncols, 1, caps, outcapdim]
    # =>[batch-size, nrows, ncols, caps, outcapdim]
    # activate: [batch-size, dims, caps, 1]
    return core.squeeze(activations.read(iterations-1), axis=-1)


def cap_conv(convop,
             bias_shape,
             logits_shape,
             iterations,
             leaky=False,
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
    """ logits_shape : [1, 1, 1, incaps, outcaps, 1]
                             for conv2d operation
    """
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             'caps'+convop.__name__,
                                             reuse)
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
    def _conv(x):
        with ops_scope:
            #  [batch-size, rows, cols, indims, incaps]
            # /*x [incapdim, outcaps, outcapdim]*/
            #=>[batch-size, nrows, ncols, dims, caps]
            x = convop(x)
            # now x is the pre-predictions denoting u^{\hat}_{j|i}
            # x shape:
            # for fully-connected:
            #     [batch-size, dims, caps, incaps]
            # for 1d:
            #     [batch-size, neurons, dims, caps, incaps]
            # for 2d:
            #     [batch-size, nrows, ncols, dims, caps, incaps]
            with core.name_scope('agreement_routing'):
                x = dynamic_routing(x,
                                    logits_shape,
                                    iterations,
                                    bias,
                                    act,
                                    leaky,
                                    epsilon,
                                    safe)
            return x
    return _conv


# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    caps_dims=int,
#                    iterations=int,
#                    leaky=bool,
#                    keepdims=bool,
#                    collections=str,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def cap_fully_connected(input_shape,
                        caps,
                        dims,
                        iterations=2,
                        leaky=False,
                        share_weights=False,
                        weight_initializer='glorot_uniform',
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
    """ fully_connected layer for capsule networks
        Attributes
        ==========
        input_shape : list / tuple
                      input tensor shape. Should in form of:
                      [batch-size, incaps, indims]
        caps : int
                output number of capsules
        dims : int
                    output capsule dimension
    """
    helper.check_input_shape(input_shape)
    batch_size = input_shape[0]
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 3:
        raise ValueError('capsule fully_connected require input shape {}[batch-size,'
                         ' incaps, incapdim]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))
    output_shape = [batch_size, caps, dims]
    incaps, indims = input_shape[-2:]
    logits_shape = [batch_size, caps, 1, incaps]
    bias_shape = [ caps, dims,1]
    if share_weights:
        weight_shape = [1, dims * caps, 1, indim]
    else:
        weight_shape = [1, dims * caps, incaps, indim]
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
    def _fully_connected(x):
        #       x: [batch-size, incaps, indims]
        #=>        [batch-size, 1, incaps, indims]
        x = core.expand_dims(x, 1)
        #       x: [batch-size, 1, incaps, indims]
        # weights: [1, dims * caps, incaps, indims]
        #=>        [batch-size, dims * caps, incaps, indims] (element-wise multiply)
        #=>        [batch-size, dims * caps, incaps] (sum along indims)
        x = core.sum(x * weights, axis=-1)
        #=>        [batch-size, caps, dims, incaps]
        return core.reshape(x, [batch_size, caps, dims, incaps])

    return cap_conv(_fully_connected,
                    bias_shape,
                    logits_shape,
                    iterations,
                    leaky,
                    bias_initializer,
                    bias_regularizer,
                    cpuid,
                    act,
                    trainable,
                    dtype,
                    epsilon,
                    safe,
                    collections,
                    summary,
                    reuse,
                    name,
                    scope), output_shape


# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    caps_dims=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    padding=str,
#                    share_weights=bool,
#                    iterations=int,
#                    collections=str,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def cap_conv1d(input_shape,
               caps,
               dims,
               iterations=3,
               leaky=False,
               kshape=3,
               stride=1,
               padding='valid',
               share_weights=False,
               weight_initializer='glorot_uniform',
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
    """ primary capsule convolutional
        Attributes
        ==========
        input_shape : list / tuple
                      should have form of:
                      [batch-size, neurons, incaps, indims]
                      where `neurons` denotes the hidden layer units
                      `incaps` denotes the vector size of each capsule
                      (as depth caps)
        caps : int
                number of output capsules
        dims : int
                    output capsule vector size (aka. output dims)
        kshape : int / list / tuple
                 kernel shape for convolving operation
    """
    raise AttributeError('function {} not implement yet'.format(__name__))
    helper.check_input_shape(input_shape)
    batch_size, neurons, incaps, indims = input_shape
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 4:
        raise ValueError('capsule conv1d require input shape {}[batch-size, '
                         'rows, cols, incaps, incapdim]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))

    kshape = helper.norm_input_2d(kshape)
    kshape[2] = indims
    stride = helper.norm_input_2d(stride)
    stride[2] = indims

    output_shape = helper.get_output_shape(input_shape, caps * dims,
                                           kshape, stride, padding)
    output_shape[0] = batch_size
    # output shape must be:
    #    [batch-size, out_neurons, caps, dims]
    output_shape = output_shape[:2] + [caps, dims]
    logits_shape = [batch_size, output_shape[1], 1, caps, incaps]
    bias_shape = [output_shape[1], caps, dims, 1]
    if share_weights:
        weight_shape = [1, caps, dims, kshape[1]*kshape[2], incaps]
    else:
        weight_shape = [output_shape[1], caps, dims, kshape[1]*kshape[2], incaps]
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
    def _conv1d(x):
        """ capsule wise convolution in 1d
        """
        # x shape:
        #     [batch-size, neurons, incaps, indims]
        #=>   [batch-size, out_neurons, 1, kr * kc * caps]
        x = core.extract_patches(x, kshape, stride, [1,1,1,1], padding.upper())
        #=>   [batch-size, out_neurons, 1, 1, kr * kc * caps]
        x = core.expand_dims(x, 3)
        #     [batch-size, out_neurons, 1, 1, kr * kc * caps]
        #=>   [batch-size, out_neurons, 1, 1, kr * kc, caps]
        x = core.reshape(x, [batch_size, output_shape[1], 1, 1, kshape[1] * kshape[2], incaps])
        #     [batch-size, out_neurons, 1, 1, kr * kc, caps]
        #=>   [batch-size, out_neurons, caps_dims, caps, kr * kc, caps] (*)
        #=>   [batch-size, out_neurons, caps_dims, caps, caps] (sum)
        x = core.sum(x * weights, axis=-2, keepdims=False)
        return x

    return cap_conv(_conv1d,
                    bias_shape,
                    logits_shape,
                    iterations,
                    leaky,
                    bias_initializer,
                    bias_regularizer,
                    cpuid,
                    act,
                    trainable,
                    dtype,
                    epsilon,
                    safe,
                    collections,
                    summary,
                    reuse,
                    name,
                    scope), output_shape


# @helpers.typecheck(input_shape=list,
#                    nouts=int,
#                    caps_dims=int,
#                    kshape=[int, list],
#                    stride=[int, list],
#                    padding=str,
#                    share_weights=bool,
#                    iterations=int,
#                    collections=str,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def cap_conv2d(input_shape,
               caps,
               dims,
               iterations=3,
               leaky=False,
               kshape=3,
               stride=1,
               padding='valid',
               share_weights=False,
               weight_initializer='glorot_uniform',
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
    """ primary capsule convolutional
        Attributes
        ==========
        input_shape : list / tuple
                      should have form of:
                      [batch-size, rows, cols, incaps, incaps_dim]
                      where `rows/cols` denotes the row/col of matrix
                      `incaps_dim` denotes the vector size of each capsule
                      (as depth channels)
                      `incaps` means the number of capsules
        nouts : int
                number of output capsules
        caps_dims : int
                    output capsule vector size (aka. outcapdim)
        kshape : int / list / tuple
                 kernel shape for convolving operation
    """
    raise AttributeError('function {} not implement yet'.format(__name__))
    helper.check_input_shape(input_shape)
    batch_size = input_shape[0]
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 5:
        raise ValueError('capsule conv2d require input shape {}[batch-size, '
                         'rows, cols, incaps, incapdim]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))
    kshape = helper.norm_input_2d(kshape)
    stride = helper.norm_input_2d(stride)
    input_nshape = input_shape[:]
    input_nshape[0] = batch_size
    #  [batch-size, rows, cols, incaps, incapdim]
    #=>[batch-size, rows, cols, incpas * incapdim]
    input_nshape[core.caxis] *= input_nshape[-2]
    input_nshape.pop(-2)
    # output shape [batch-size, nrows, ncols, nouts, caps_dims]
    output_shape = helper.get_output_shape(input_nshape, caps * dims,
                                           kshape, stride, padding)
    output_shape[0] = batch_size
    output_shape = output_shape[:-1] + [caps, dims]
    batch_size, rows, cols, incaps, indims = input_shape
    logits_shape = output_shape[:3] + [incaps, caps, 1]
    bias_shape = [caps, dims]
    if share_weights:
        # share filter for capsules along incaps
        # kernel shape:
        # [krow, kcol, incapdims, outcaps * outcapdims]
        kernel_shape = kshape[1:-1] + [indims, caps * dims]
        weights = mm.malloc('weights',
                            helper.normalize_name(name),
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
        def _conv2d(x):
            #  [batch-size, rows, cols, incaps, incapdim]
            #=>[batch-size, incaps, rows, cols, incapdim]
            x = core.transpose(x, (0, 4, 1, 2, 3))
            #  [batch-size, incaps, rows, cols, incapdim]
            #=>[batch-size * incaps, rows, cols, incapdim]
            x = core.reshape(x, (-1, rows, cols, indims))
            x = core.conv2d(x, weights, stride, padding)
            #  [batch-size * incaps, nrows, ncols, outcaps * caps_dims]
            #=>[batch-size, incaps, nrows, ncols, outcaps, caps_dims]
            x = core.reshape(x, [-1, incaps] + output_shape[1:])
            #  [batch-size, incaps, nrows, ncols, outcaps, caps_dims]
            #=>[batch-size, nrows, ncols, incaps, outcaps, caps_dims]
            return core.transpose(x, (0, 2, 3, 1, 4, 5))
    else:
        # kernel shape:
        # [krow, kcol, incaps, incapdims, outcaps * outcapdims]
        kernel_shape = kshape[1:-1] + input_shape[-2:] + [caps * dims]
        weights = mm.malloc('weights',
                            helper.normalize_name(name),
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
        def _body(idx, x, array):
            # kernels shape : [krow, kcol, incaps, incapdims, outcaps * outcapdims]
            # kernel shape  : [krow, kcol, incapdims, outcaps * outcapdims]
            weight = core.gather(weights, idx, axis=-3)
            # x shape    : [batch-size, rows, cols, incaps, incapdim]
            # subx shape : [batch-size, rows, cols, incapdim]
            subx = core.gather(x, idx, axis=-2)
            conv2d_output = core.conv2d(subx,
                                        weight,
                                        stride,
                                        padding.upper())
            array = array.write(idx, conv2d_output)
            return [idx + 1, x, array]

        def _conv2d(x):
            """ capsule wise convolution in 2d
                that is, convolve along `incaps` dimension
            """
            # x shape:
            #     [batch-size, rows, cols, incaps, incapdim]
            iterations = input_shape[-2] # <- incaps
            idx = core.constant(0, core.int32)
            array = core.TensorArray(dtype=core.float32,
                                     size=iterations)
            _, x, array = core.while_loop(
                lambda idx, x, array : idx < iterations,
                _body,
                loop_vars = [idx, x, array],
                parallel_iterations=iterations
            )
            # array should have the shape of:
            # incaps * [batch-size, nrows, ncols, outcaps * outcapdims]
            # stack to
            # [incaps, batch-size, nrows, ncols, outcaps * outcapdims]
            array = array.stack()
            # then reshape to
            # [incaps, batch-size, nrows, ncols, outcaps, outcapdims]
            newshape = [iterations] + core.shape(array)[1:-1] + [caps, dims]
            array = core.reshape(array, newshape)
            # then transpose to
            # [batch-size, nrows, ncols, incaps, outcaps, caps_dims]
            array = core.transpose(array, (1, 2, 3, 0, 4, 5))
            return array

    return cap_conv(_conv2d,
                    bias_shape,
                    logits_shape,
                    iterations,
                    leaky,
                    bias_initializer,
                    bias_regularizer,
                    cpuid,
                    act,
                    trainable,
                    dtype,
                    epsilon,
                    safe,
                    collections,
                    summary,
                    reuse,
                    name,
                    scope), output_shape
