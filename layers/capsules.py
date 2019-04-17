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
def norm(inputs,
         axis=None,
         keepdims=False,
         ord='euclidean',
         epsilon=None,
         safe=False,
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
    fun, output = ops.capsules.norm(input_shape,
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
def fully_connected(inputs, nouts, caps_dims,
                    iterations=2,
                    leaky=False,
                    share_weights=True,
                    weight_initializer='glorot_uniform',
                    weight_regularizer=None,
                    bias_initializer='zeros', # no bias
                    bias_regularizer=None,
                    cpuid=0,
                    act=None,
                    trainable=True,
                    dtype=ops.core.float32,
                    return_shape=False,
                    check_shape=True,
                    epsilon=ops.core.epsilon,
                    safe=False,
                    collections=None,
                    summary='histogram',
                    reuse=False,
                    name=None,
                    scope=None):
    """ fully_connected operation between capsules
    """
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.fully_connected(input_shape,
                                               nouts,
                                               caps_dims,
                                               iterations,
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


dense = fully_connected


@core.layer
def conv1d(inputs, nouts, caps_dims, kshape,
           iterations=3,
           leaky=False,
           stride=1,
           padding='valid',
           share_weights=True,
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           cpuid=0,
           act=None,
           trainable=True,
           dtype=ops.core.float32,
           return_shape=False,
           check_shape=True,
           epsilon=ops.core.epsilon,
           safe=False,
           collections=None,
           summary='histogram',
           reuse=False,
           name=None,
           scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.conv1d(input_shape,
                                      nouts,
                                      caps_dims,
                                      kshape,
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
                                      reuse,
                                      name,
                                      scope)
    return convs._layers(fun, inputs, output, return_shape, check_shape)


@core.layer
def conv2d(inputs, nouts, caps_dims, kshape,
           iterations=3,
           leaky=False,
           stride=1,
           padding='valid',
           share_weights=True,
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           cpuid=0,
           act=None,
           trainable=True,
           dtype=ops.core.float32,
           return_shape=False,
           check_shape=True,
           epsilon=ops.core.epsilon,
           safe=False,
           collections=None,
           summary='histogram',
           reuse=False,
           name=None,
           scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.capsules.conv2d(input_shape,
                                      nouts,
                                      caps_dims,
                                      kshape,
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
                                      reuse,
                                      name,
                                      scope)
    return convs._layers(fun, inputs, output, return_shape, check_shape)
