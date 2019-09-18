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

from sigma import colors, helpers
from .. import core, actives, helper, mm
import logging

from . import capsules_dc, capsules_cd


# @helpers.typecheck(input_shape=list,
#                    axis=int,
#                    drop=bool,
#                    flatten=bool,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def cap_maskout(input_shape,
            order='DC',
            onehot=True,
            drop=False,
            flatten=True,
            reuse=False,
            name=None,
            scope=None):
    """ typical input_shape form:
            [batch-size, dims, caps]
        or[batch-size, caps, dims]
        axis: axis of caps
        flatten works ONLY when drop is `False`
    """
    fun = capsules_dc.cap_maskout
    if order == 'CD':
        fun = capsules_cd.cap_maskout
    return fun(input_shape,
               onehot,
               drop,
               flatten,
               reuse,
               name,
               scope)

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
             axis,
             keepdims=False,
             ord='euclidean',
             epsilon=core.epsilon,
             safe=True,
             act=None,
             reuse=None,
             name=None,
             scope=None):
    """ classically,  inputs_shape is in the form of
        [batch-size, dims, channels]
        this function calculates the norm of each capsule
        along capdims dimension
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
            #   [batch-size, rows, cols, dims, channels]
            # =>[batch-size, rows, cols, channels]
            return act(core.norm(x, axis, keepdims, ord, epsilon, safe))
    return _norm, output_shape

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
                        iterations=3,
                        order='DC',
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
                      [batch-size, incapdim, channels]
        nouts : int
                output number of capsules
        caps_dims : int
                    output capsule dimension
    """
    fun = capsules_dc.cap_fully_connected
    if order == 'CD':
        fun = capsules_cd.cap_fully_connected
    return fun(input_shape,
               caps,
               dims,
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
    fun = capsules_dc.cap_conv1d
    if order == 'CD':
        fun = capsules_cd.cap_conv1d
    return fun(input_shape,
               caps,
               dims,
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
    fun = capsules_dc.cap_conv2d
    if order == 'CD':
        fun = capsules_cd.cap_conv2d
    return fun(input_shape,
               caps,
               dims,
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
