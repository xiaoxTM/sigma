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

from sigma import colors
from . import capsules_cd
from . import capsules_dc
from .. import core, helper, actives

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



# @helpers.typecheck(input_shape=list,
#                    axis=int,
#                    keepdims=bool,
#                    ord=str,
#                    epsilon=float,
#                    safe=bool,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
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
                              check_input_shape=True,
                              reuse=None,
                              name=None,
                              scope=None):
    """ classically,  inputs_shape is in the form of
        [batch-size, channels, dims]
        this function calculates the norm of each capsule
        along capsdims dimension
    """
    if check_input_shape:
        helper.check_input_shape(input_shape)
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    ops_scope, _, _ = helper.assign_scope(name, scope, 'caps_norm', reuse)
    if axis is None:
        axis = -1
    act = actives.get(act)
    output_shape = input_shape[:]
    output_shape.pop(axis)
    def _norm(x):
        with ops_scope:
            # the length (norm) of the activity vector of each
            # capsule in digit_caps layer indicates presence
            # of an instance of each class
            #   [batch-size, rows, cols, channels, dims]
            # =>[batch-size, rows, cols, channels]
            return act(core.norm(x, axis, keepdims, ord, epsilon, safe))
    return _norm, output_shape

# @helpers.typecheck(input_shape=list,
#                    channels=int,
#                    dims=int,
#                    iterations=int,
#                    leaky=bool,
#                    keepdims=bool,
#                    collections=str,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def cap_fully_connected(input_shape,
                        channels,
                        dims,
                        iterations=2,
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
                        check_input_shape=True,
                        reuse=False,
                        name=None,
                        scope=None):
    """ fully_connected layer for capsule networks
        Attributes
        ==========
        input_shape : list / tuple
                      input tensor shape. Should in form of:
                      [batch-size, incapdim, channels]
        channels : int
                output number of capsules
        dims : int
                    output capsule dimension
    """
    if order not in {'DC', 'CD'}:
        raise ValueError('`order` must be one of {}/{}'.format(colors.red('DC'), colors.red('CD')))
    _fully_connected = capsules_dc.cap_fully_connected
    if order == 'CD':
        _fully_connected = capsules_cd.cap_fully_connected
    return _fully_connected(input_shape,
                            channels,
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
                            check_input_shape,
                            reuse,
                            name,
                            scope)

""" capsule projection / transformation operation
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
def cap_project(input_shape,
                channels,
                dims,
                order='DC',
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
    if order not in {'DC', 'CD'}:
        raise ValueError('`order` must be one of {}/{}'.format(colors.red('DC'), colors.red('CD')))
    _project= capsules_dc.cap_project
    if order == 'CD':
        _project = capsules_cd.cap_project
    return _project(input_shape,
                    channels,
                    dims,
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

""" permutation transformation operation
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
def cap_permutation_transform(input_shape,
                              channels,
                              dims,
                              order='DC',
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
    if order not in {'DC', 'CD'}:
        raise ValueError('`order` must be one of {}/{}'.format(colors.red('DC'), colors.red('CD')))
    _permutation_transform = capsules_dc.cap_permutation_transform
    if order == 'CD':
        _permutation_transform = capsules_cd.cap_permutation_transform
    return _permutation_transform(input_shape,
                                  channels,
                                  dims,
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

# @helpers.typecheck(input_shape=list,
#                    channels=int,
#                    dims=int,
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
               dtype=core.float32,
               epsilon=core.epsilon,
               safe=True,
               collections=None,
               summary='histogram',
               check_input_shape=True,
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
        channels : int
                number of output capsules
        dims : int
                    output capsule vector size (aka. outcapdim)
        kshape : int / list / tuple
                 kernel shape for convolving operation
    """
    if order not in {'DC', 'CD'}:
        raise ValueError('`order` must be one of {}/{}'.format(colors.red('DC'), colors.red('CD')))
    _cap_conv1d = capsules_dc.cap_conv1d
    if order == 'CD':
	    raise ValueError('`order` not support {} currently. please use `DC` after transpose instead'
		                 .format(colors.red(order)))
        #_cap_conv1d = capsules_cd.cap_conv1d
    return _cap_conv1d(input_shape,
                       channels,
                       dims,
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
                       check_input_shape,
                       reuse,
                       name,
                       scope)


# @helpers.typecheck(input_shape=list,
#                    channels=int,
#                    dims=int,
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
               dtype=core.float32,
               epsilon=core.epsilon,
               safe=True,
               collections=None,
               summary='histogram',
               check_input_shape=True,
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
        channels : int
                number of output capsules
        dims : int
                    output capsule vector size (aka. outcapdim)
        kshape : int / list / tuple
                 kernel shape for convolving operation
    """
    if order not in {'DC', 'CD'}:
        raise ValueError('`order` must be one of {}/{}'.format(colors.red('DC'), colors.red('CD')))
    _cap_conv2d = capsules_dc.cap_conv2d
    if order == 'CD':
	    raise ValueError('`order` not support {} currently. please use `DC` after transpose instead'
		                 .format(colors.red(order)))
        #_cap_conv2d = capsules_cd.cap_conv2d
    return _cap_conv2d(input_shape,
                       channels,
                       dims,
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
                       cat,
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
