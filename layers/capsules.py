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
def cap_norm(inputs,
                              axis,
                              keepdims=False,
                              ord='euclidean',
                              epsilon=ops.core.epsilon,
                              safe=True,
                              act=None,
                              return_shape=False,
                              check_output_shape=True,
                              check_input_shape=True,
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
                                        order,
                                        ord,
                                        epsilon,
                                        safe,
                                        act,
                                        check_input_shape,
                                        reuse,
                                        name,
                                        scope)
    return convs._layers(fun, inputs, output, return_shape, check_output_shape)


@core.layer
def cap_fully_connected(inputs,
                        channels,
                        dims,
                        iterations=2,
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
                        epsilon=ops.core.epsilon,
                        safe=True,
                        collections=None,
                        summary='histogram',
                        check_output_shape=True,
                        check_input_shape=True,
                        reuse=False,
                        name=None,
                        scope=None):
    """ fully_connected operation between capsules
    """
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.cap_fully_connected(input_shape,
                                                   channels,
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
                                                   check_input_shape,
                                                   reuse,
                                                   name,
                                                   scope)
    return convs._layers(fun, inputs, output, return_shape, check_output_shape)


@core.layer
def permutation_transform(inputs,
                          channels,
                          dims,
                          order='DC',
                          mode='max',
                          weight_initializer=core.__defaults__['weight_initializer'],
                          weight_regularizer=core.__defaults__['weight_regularizer'],
                          bias_initializer=core.__defaults__['bias_initializer'],
                          bias_regularizer=core.__defaults__['bias_regularizer'],
                          cpuid=0,
                          act=None,
                          trainable=True,
                          dtype=ops.core.float32,
                          return_shape=False,
                          collections=None,
                          summary=core.__defaults__['summary'],
                          check_output_shape=True,
                          check_input_shape=True,
                          reuse=False,
                          name=None,
                          scope=None):
    ''' order invariance transform layer
        i.e., the output feature indepenedent of input order
        inputs should have shape of
        [batch-size, indims, incaps]
    '''
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.order_invariance_transform(input_shape,
                                                          channels,
                                                          dims,
                                                          order,
                                                          mode,
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
                                                          check_input_shape,
                                                          reuse,
                                                          name,
                                                          scope)
    return convs._layers(fun, inputs, output, return_shape, check_output_shape)


@core.layer
def cap_conv1d(inputs,
               channels,
               dims,
               kshape,
               iterations=3,
               order='DC',
               leaky=False,
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
               epsilon=ops.core.epsilon,
               safe=True,
               collections=None,
               summary='histogram',
               check_output_shape=True,
               check_input_shape=True,
               reuse=False,
               name=None,
               scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.cap_conv1d(input_shape,
                                          channels,
                                          dims,
                                          kshape,
                                          order,
                                          iterations,
                                          leaky,
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
                                          check_input_shape,
                                          reuse,
                                          name,
                                          scope)
    return convs._layers(fun, inputs, output, return_shape, check_output_shape)


@core.layer
def cap_conv2d(inputs,
               channels,
               dims,
               kshape,
               iterations=3,
               order='DC',
               leaky=False,
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
               epsilon=ops.core.epsilon,
               safe=True,
               collections=None,
               summary='histogram',
               check_output_shape=True,
               check_input_shape=True,
               reuse=False,
               name=None,
               scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.cap_conv2d(input_shape,
                                          channels,
                                          dims,
                                          kshape,
                                          iterations,
                                          order,
                                          leaky,
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
                                          check_input_shape,
                                          reuse,
                                          name,
                                          scope)
    return convs._layers(fun, inputs, output, return_shape, check_output_shape)

# alien name
norm = cap_norm
fully_connected = cap_fully_connected
dense = cap_fully_connected
conv1d = cap_conv1d
conv2d = cap_conv2d
