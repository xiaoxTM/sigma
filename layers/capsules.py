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

from .. import ops, colors
from . import core, convs

@core.layer
def cap_maskout(inputs,
            index,
            order='DC',
            onehot=True,
            drop=False,
            flatten=True,
            return_shape=False,
            reuse=False,
            name=None,
            scope=None):
    """ maskout specificated features given by `indices`
        NOTE this layer will drop if `drop` is True the other indices
        > inputs : [batch-size, dims=feature length, channels]
        > outputs: [batch-size, len(indices), channels] if indices
          have more than one indices
        > outputs: [batch-size, channels] if indices have only one indices
    """
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.cap_maskout(input_shape,
                                   order,
                                   onehot,
                                   drop,
                                   flatten,
                                   reuse,
                                   name,
                                   scope)
    x = fun(inputs,index)
    xshape = ops.core.shape(x)
    if output[1:] != xshape[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(colors.red(output),
                                 colors.green(xshape)))
    if return_shape:
        x = [x, output]
    return x


@core.layer
def cap_norm(inputs,
             axis,
             keepdims=False,
             ord='euclidean',
             epsilon=ops.core.epsilon,
             safe=True,
             act=None,
             return_shape=False,
             check_shape=True,
             reuse=False,
             name=None,
             scope=None):
    """ norm of vector
        input vector output scalar
    """
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.cap_norm(input_shape,
                                        axis,
                                        keepdims,
                                        ord,
                                        epsilon,
                                        safe,
                                        act,
                                        reuse,
                                        name,
                                        scope)
    return convs._layers(fun, inputs, output, return_shape, check_shape)


@core.layer
def cap_fully_connected(inputs,
                        caps,
                        dims,
                        iterations=3,
                        order='DC',
                        leaky=False,
                        share_weights=False,
                        weight_initializer='glorot_uniform',
                        weight_regularizer=None,
                        bias_initializer='zeros', # no bias
                        bias_regularizer=None,
                        cpuid=0,
                        act='squash',
                        trainable=True,
                        dtype=ops.core.float32,
                        return_shape=False,
                        check_shape=True,
                        epsilon=ops.core.epsilon,
                        safe=True,
                        collections=None,
                        summary='histogram',
                        reuse=False,
                        name=None,
                        scope=None):
    """ fully_connected operation between capsules
    """
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.cap_fully_connected(input_shape,
                                                   caps,
                                                   dims,
                                                   iterations,
                                                   order,
                                                   leaky,
                                                   share_weights,
                                                   weight_initializer,
                                                   weight_regularizer,
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
                                                   scope)
    return convs._layers(fun, inputs, output, return_shape, check_shape)


@core.layer
def cap_conv1d(inputs,
               caps,
               dims,
               order='DC',
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
               dtype=ops.core.float32,
               return_shape=False,
               check_shape=True,
               epsilon=ops.core.epsilon,
               safe=True,
               collections=None,
               summary='histogram',
               reuse=False,
               name=None,
               scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.cap_conv1d(input_shape,
                                          caps,
                                          dims,
                                          order,
                                          iterations,
                                          leaky,
                                          kshape,
                                          stride,
                                          padding,
                                          share_weights,
                                          weight_initializer,
                                          weight_regularizer,
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
                                          scope)
    return convs._layers(fun, inputs, output, return_shape, check_shape)


@core.layer
def cap_conv2d(inputs, caps, dims,
               order='DC',
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
               dtype=ops.core.float32,
               return_shape=False,
               check_shape=True,
               epsilon=ops.core.epsilon,
               safe=True,
               collections=None,
               summary='histogram',
               reuse=False,
               name=None,
               scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.cap_conv2d(input_shape,
                                          caps,
                                          dims,
                                          order,
                                          iterations,
                                          leaky,
                                          kshape,
                                          stride,
                                          padding,
                                          share_weights,
                                          weight_initializer,
                                          weight_regularizer,
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
                                          scope)
    return convs._layers(fun, inputs, output, return_shape, check_shape)

# alien name
maskout = cap_maskout
norm = cap_norm
fully_connected = cap_fully_connected
dense = cap_fully_connected
conv1d = cap_conv1d
conv2d = cap_conv2d
