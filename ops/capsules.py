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
#           [batch-size, capdim, channels]
#vs normal: [batch-size, channels]
#       2> for conv1d capsules:
#           [batch-size, neurons, capdim, channels]
#vs normal: [batch-size, neurons, channels]
#       3> for conv2d capsules:
#           [batch-size, rows, cols, capdim, channels]
#vs normal: [batch-size, rows, cols, channels]


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
    nouts = shape[core.axis]
    shape[core.axis] = 1
    leak = core.zeros(shape)
    # add an extra dimension to routes
    leaky_logits = core.concat([leak, logits], axis=core.axis)
    # routes to capsules who has max probability
    leaky_routs = core.softmax(leaky_logits, axis=core.axis)
    # remove added dimension
    return core.split(leaky_routs, [1, nouts], axis=core.axis)[1]


def _agreement_routing(prediction,
                       logits_shape,
                       iterations,
                       bias, #[channels, 1]
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
                     [batch-size, outcapdim, outcaps, incaps]
                     for fully_connected, and
                     [batch-size, neurons, outcapdim, outcaps, incaps]
                     for conv1d, and
                     [batch-size, rows, cols, outcapdim, outcaps, incaps]
                     for conv2d
        logits_shape : Tensor
                       denotes b_{i,j} in paper
                       with the shape of
                       [batch-size, 1, outcaps, incaps]
                       for fully_connected, and
                       [batch-size, 1, 1, outcaps, incaps]
                       for conv1d, and
                       [batch-size, 1, 1, 1, outcaps, incaps]
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
        [batch-size, outcapdims, outcaps]
        for fully_connected
        [batch-size, neurons, outcapdims, outcaps]
        for conv1d
        [batch-size, rows, cols, outcapdims, outcaps]
        for conv2d
    """
    # restore v_j
    activations = core.TensorArray(dtype=core.float32,
                                   size=iterations,
                                   clear_after_read=False)
    logits = core.zeros(logits_shape, dtype=core.float32)
    act = actives.squash(-2, epsilon, safe, aslayer=False)
    idx = core.constant(0, dtype=core.int32)
    # softmax along with `outcaps` axis

    no_grad_prediction = core.stop_gradient(prediction, name='stop_gradient')

    def _update(i, logits, activations):
        """ dynamic routing to update coefficiences (c_{i, j})
            logits : [batch-size, /*rows, cols,*/ incaps, outcaps, 1]
        """
        # softmax along with `outcaps` axis
        # that is, `agree` on some capsule
        # a.k.a., which capsule in higher layer to activate
        #        logits: [batch-size, 1, outcaps, incaps]
        #              / [batch-size, neurons, 1, outcaps, incaps]
        #              / [batch-size, rows, cols, 1, outcaps, incaps]
        #=>coefficients: [batch-size, 1, outcaps, incaps]
        #              / [batch-size, neurons, 1, outcaps, incaps]
        #              / [batch-size, rows, cols, 1, outcaps, incaps]
        coefficients = core.softmax(logits, axis=-2)
        # average all lower capsules's prediction to higher capsule
        # e.g., sum along with `incaps` axis
        # that is, agreements from all capsules in lower layer
        if i == iterations - 1:
            # bias: [outdims, outcaps, 1]
            # coefficients * prediction:
            #=> [batch-size, outdims, outcaps, incaps] (i.e., elements in the same capsules shares bias)
            # sum operation (i.e., preactivate):
            #=> [batch-size, outdims, outcaps, 1]
            preactivate = core.sum(coefficients * prediction,
                                   axis=-1,
                                   keepdims=True) + bias
            activation = act(preactivate)
            activations = activations.write(i, activation)
        else:
            preactivate = core.sum(coefficients * no_grad_prediction,
                                   axis=-1,
                                   keepdims=True) + bias
            # typically, squash
            activation = act(preactivate)
            activations = activations.write(i, activation)
            # sum up along `outcapdim`
            # prediction * activation:
            #=> [batch-size, outdims, outcaps, incaps]
            # * [batch-size, outdims, outcaps, 1]
            #=> [batch-size, outdims, outcaps, incaps]
            # sum:
            #=> [batch-size, 1, outcaps, incaps]
            distance = core.sum(prediction * activation,
                                axis=-3,
                                keepdims=True)
            logits += distance
        return (i+1, logits, activations)

    _, logits, activations = core.while_loop(
        lambda idx, logits, activations: idx < iterations,
        _update,
        loop_vars=[idx, logits, activations],
        swap_memory=True)
    #   [batch-size, nrows, ncols, 1, outcaps, outcapdim]
    # =>[batch-size, nrows, ncols, outcaps, outcapdim]
    # activate: [batch-size, outdims, outcaps, 1]
    return core.squeeze(activations.read(iterations-1), axis=-1)


# @helpers.typecheck(input_shape=list,
#                    axis=int,
#                    keepdims=bool,
#                    ord=str,
#                    epsilon=float,
#                    safe=bool,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def cap_norm(input_shape,
             axis=None,
             keepdims=False,
             ord='euclidean',
             epsilon=core.epsilon,
             safe=True,
             act=None,
             reuse=None,
             name=None,
             scope=None):
    """ classically,  inputs_shape is in the form of
        [batch-size, capdims, channels]
        this function calculates the norm of each capsule
        along capsdims dimension
    """
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    ops_scope, _, _ = helper.assign_scope(name, scope, 'caps_norm', reuse)
    if axis is None:
        axis = -2
    act = actives.get(act)
    output_shape = input_shape[:]
    output_shape.pop(axis)
    def _norm(x):
        with ops_scope:
            # the length (norm) of the activity vector of each
            # capsule in digit_caps layer indicates presence
            # of an instance of each class
            #   [batch-size, rows, cols, capdim, channels]
            # =>[batch-size, rows, cols, channels]
            return act(core.norm(x, axis, keepdims, ord, epsilon, safe))
    return _norm, output_shape


def cap_conv(convop, bias_shape, logits_shape, iterations,
             leaky=False,
             bias_initializer='zeros',
             bias_regularizer=None,
             cpuid=0,
             act=None,
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
    act = actives.get(act)
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
            #  [batch-size, rows, cols, incaps, incapdim]
            # /*x [incapdim, outcaps, outcapdim]*/
            #=>[batch-size, nrows, ncols, incaps, outcaps, outcapdim]
            x = act(convop(x))
            # now x is the pre-predictions denoting u^{\hat}_{j|i}
            # x shape:
            # for fully-connected:
            #     [batch-size, incaps, outcaps, caps_dims]
            # for 1d:
            #     [batch-size, neurons, incaps, outcaps, caps_dims]
            # for 2d:
            #     [batch-size, nrows, ncols, incaps, outcaps, caps_dims]
            with core.name_scope('agreement_routing'):
                x = _agreement_routing(x, logits_shape, iterations,
                                       bias, leaky, epsilon, safe)
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
def cap_fully_connected(input_shape, channels, caps_dims,
                        iterations=2,
                        leaky=False,
                        share_weights=False,
                        weight_initializer='glorot_uniform',
                        weight_regularizer=None,
                        bias_initializer='zeros',
                        bias_regularizer=None,
                        cpuid=0,
                        act=None,
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
                      [batch-size, incapdim, channels]
        nouts : int
                output number of capsules
        caps_dims : int
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
    output_shape = [batch_size, caps_dims, channels]
    incapdim, incaps = input_shape[-2:]
    logits_shape = [batch_size, 1, channels, incaps]
    bias_shape = [caps_dims, channels, 1]
    if share_weights:
        weight_shape = [1, incapdim, caps_dims * channels, 1]
    else:
        weight_shape = [1, incapdim, caps_dims * channels, incaps]
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
    def _fully_connected(x):
        #       x: [batch-size, indims, inchannels=incaps]
        #=>        [batch-size, indims, 1, inchannels]
        x = core.expand_dims(x, 2)
        #       x: [batch-size, indims, 1, incaps]
        # weights: [1, indims, outdims * channels, incaps]
        #=>        [batch-size, indims, outdims * channels, incaps] (element-wise multiply)
        #=>        [batch-size, outdims * channels, incaps] (sum along indims)
        x = core.sum(x * weights, axis=1)
        #=>        [batch-size, outdims, channels, incaps]
        return core.reshape(x, [-1, caps_dims, channels, incaps])

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
               channels,
               caps_dims,
               kshape,
               iterations=3,
               leaky=False,
               stride=1,
               padding='valid',
               share_weights=False,
               weight_initializer='glorot_uniform',
               weight_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               cpuid=0,
               act=None,
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
                      [batch-size, neurons, incaps_dim, incaps=inchannels]
                      where `neurons` denotes the hidden layer units
                      `incaps` denotes the vector size of each capsule
                      (as depth channels)
        nouts : int
                number of output capsules
        caps_dims : int
                    output capsule vector size (aka. outcapdim)
        kshape : int / list / tuple
                 kernel shape for convolving operation
    """
    helper.check_input_shape(input_shape)
    batch_size, neurons, incap_dims, inchannels = input_shape
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 4:
        raise ValueError('capsule conv1d require input shape {}[batch-size, '
                         'rows, cols, incaps, incapdim]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))

    kshape = helper.norm_input_2d(kshape)
    kshape[2] = incap_dims
    stride = helper.norm_input_2d(stride)
    stride[2] = incap_dims

    input_nshape = input_shape[:]
    input_nshape[0] = batch_size
    #  [batch-size, neurons, incapdim, incaps]
    #=>[batch-size, neurons, incpas * incapdim]
    # //FUTURE: remove hard-coding of number of `-2`
    input_nshape[core.axis] *= input_nshape[-2]
    input_nshape.pop(-2)
    # output shape may be not right
    output_shape = helper.get_output_shape(input_nshape, channels * caps_dims,
                                           kshape, stride, padding)
    output_shape[0] = batch_size
    # output shape must be:
    #    [batch-size, out_neurons, caps_dims, channels]
    output_shape = output_shape[:-1] + [caps_dims, channels]
    logits_shape = [batch_size, output_shape[1], 1, channels, inchannels]
    bias_shape = [output_shape[1], caps_dims, channels, 1]
    if share_weights:
        weight_shape = [1, caps_dims, channels, kshape[1]*kshape[2], inchannels]
    else:
        weight_shape = [neurons, caps_dims, channels, kshape[1]*kshape[2], inchannels]
    weights = mm.malloc('weights',
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
    def _conv1d(x):
        """ capsule wise convolution in 1d
        """
        # x shape:
        #     [batch-size, neurons, incapdim, incaps]
        #=>   [batch-size, out_neurons, 1, kr * kc * incaps]
        x = core.extract_patches(x, kshape, stride, [1,1,1,1], padding)
        #=>   [batch-size, out_neurons, 1, 1, kr * kc * incaps]
        x = core.expand_dims(x, 3)
        #     [batch-size, out_neurons, 1, 1, kr * kc * incaps]
        #=>   [batch-size, out_neurons, 1, 1, kr * kc, incaps]
        x = core.reshape(x, [batch-size, 1, 1, kshape[0] * kshape[1], inchannels])
        #     [batch-size, out_neurons, 1, 1, kr * kc, incaps]
        #=>   [batch-size, out_neurons, caps_dims, channels, kr * kc, incaps]
        #=>   [batch-size, out_neurons, caps_dims, channels, incaps]
        return core.sum(x * weights, axis=-2, keepdims=False)

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
def cap_conv2d(input_shape, nouts, caps_dims, kshape,
               iterations=3,
               leaky=False,
               stride=1,
               padding='valid',
               share_weights=False,
               weight_initializer='glorot_uniform',
               weight_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               cpuid=0,
               act=None,
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
    input_nshape[core.axis] *= input_nshape[-2]
    input_nshape.pop(-2)
    # output shape [batch-size, nrows, ncols, nouts, caps_dims]
    output_shape = helper.get_output_shape(input_nshape, nouts * caps_dims,
                                           kshape, stride, padding)
    output_shape[0] = batch_size
    output_shape = output_shape[:-1] + [nouts, caps_dims]
    batch_size, rows, cols, incaps, incapdim = input_shape
    logits_shape = output_shape[:3] + [incaps, nouts, 1]
    bias_shape = [nouts, caps_dims]
    if share_weights:
        # share filter for capsules along incaps
        # kernel shape:
        # [krow, kcol, incapdims, outcaps * outcapdims]
        kernel_shape = kshape[1:-1] + [incapdim, nouts * caps_dims]
        weights = mm.malloc('weights',
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
        def _conv2d(x):
            #  [batch-size, rows, cols, incaps, incapdim]
            #=>[batch-size, incaps, rows, cols, incapdim]
            x = core.transpose(x, (0, 4, 1, 2, 3))
            #  [batch-size, incaps, rows, cols, incapdim]
            #=>[batch-size * incaps, rows, cols, incapdim]
            x = core.reshape(x, (-1, rows, cols, incapdim))
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
        kernel_shape = kshape[1:-1] + input_shape[-2:] + [nouts * caps_dims]
        weights = mm.malloc('weights',
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
            newshape = [iterations] + core.shape(array)[1:-1] + [nouts, caps_dims]
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
