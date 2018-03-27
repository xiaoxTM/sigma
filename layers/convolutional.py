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
from . import core


def _layers(fun, inputs, output, return_shape, typename, reuse, name):
    x = fun(inputs)
    if output != ops.core.shape(x):
        raise ValueError('the predicted output shape and the real'
                         ' output shape not match. {} vs {}'
                         .format(colors.green(output),
                                 colors.red(ops.core.shape(x))))
    if return_shape:
        x = [x, output]
    return x


""" convolutional layers

    Attribute
    =========
        scope : string
                scope of weights / biases. can be used to retrieve them
        inputs : tensor
                 input tensor
        nouts : int
                number of output channels
        weight_initializer : string | callable function | None
                             initializer to initialize weights when allocate memory
        weight_regularizer : string | None
                             when not None, will be used to calculate with weight
                             penalty for loss to prevent overfitting
        bias_initializer : string | callable function | None
                           initializer to initialize bias when allocate memory
        weight_regularizer : string | None
                             when not None, will be used to calculate with bias
                             penalty for loss to prevent overfitting
        act : string | callable | None
              activation function
        trainable : bool
                    indicates whether weight and bias can be trained or not
                    when back-propagating
        dtype : tensorflow.dtype
                data type of allocated weight / bias
        collections : string | None
                      name of collections. if None, default is GLOBAL
                      (therefore can be initialized by call:
                       ```python
                          session.run(global_variables_initializer())
                       ```
                      )
        summary : string / None [`histogram`, `image`, `text`, `scalar`]
                  if not None, write to summary as `summary`
        reuse : bool
                whether should reuse weight / bias instead of allocat new
        name : string | None
               variable name when allocating one
"""

""" fully connected operation
"""
@core.layer
def fully_connected(inputs, nouts,
                    weight_initializer=core.__defaults__['weight_initializer'],
                    weight_regularizer=core.__defaults__['weight_regularizer'],
                    bias_initializer=core.__defaults__['bias_initializer'],
                    bias_regularizer=core.__defaults__['bias_regularizer'],
                    act=None,
                    trainable=True,
                    dtype=ops.core.float32,
                    return_shape=False,
                    collections=None,
                    summary=core.__defaults__['summary'],
                    reuse=False,
                    name=None,
                    scope=None):
    input_shape = ops.core.shape(inputs)
    fun, output = ops.convs.fully_connected(input_shape,
                                            nouts,
                                            weight_initializer,
                                            weight_regularizer,
                                            bias_initializer,
                                            bias_regularizer,
                                            act,
                                            trainable,
                                            dtype,
                                            collections,
                                            summary,
                                            reuse,
                                            name,
                                            scope)
    return _layers(fun, inputs, output, return_shape,
                   'fully-connected', reuse, name)

# alias of fully-connected
dense = fully_connected

""" 1-D convolutional operation
"""
@core.layer
def conv1d(inputs, nouts,
           kshape=3,
           stride=1,
           padding=core.__defaults__['padding'],
           weight_initializer=core.__defaults__['weight_initializer'],
           weight_regularizer=core.__defaults__['weight_regularizer'],
           bias_initializer=core.__defaults__['bias_initializer'],
           bias_regularizer=core.__defaults__['bias_regularizer'],
           act=None,
           trainable=True,
           dtype=ops.core.float32,
           return_shape=False,
           collections=None,
           summary=core.__defaults__['summary'],
           reuse=False,
           name=None,
           scope=None):
    input_shape = ops.core.shape(inputs)
    fun, output = ops.convs.conv1d(input_shape,
                                   nouts,
                                   kshape,
                                   stride,
                                   padding,
                                   weight_initializer,
                                   weight_regularizer,
                                   bias_initializer,
                                   bias_regularizer,
                                   act,
                                   trainable,
                                   dtype,
                                   collections,
                                   summary,
                                   reuse,
                                   name,
                                   scope)

    return _layers(fun, inputs, output, return_shape,
                   'conv1d', reuse, name)


# @layer
# def soft_conv1d(inputs, nouts, kshape, stride, padding='valid',
#                 offsets=None, offsets_trainable=False,
#                 weight_initializer='glorot_uniform',
#                 weight_regularizer=None,
#                 bias_initializer='zeros',
#                 bias_regularizer=None,
#                 act=None, trainable=True,
#                 dtype=core.float32,
#                 collections=None,
#                 reuse=False, summary='histogram',
#                 name=None, scope=None):
#     input_shape = inputs.get_shape().as_list()
#     fun, output = convs.soft_conv1d(input_shape, nouts,
#                                     kshape, stride, padding,
#                                     weight_initializer,
#                                     weight_regularizer,
#                                     bias_initializer,
#                                     bias_regularizer,
#                                     act, trainable, dtype,
#                                     collections, reuse,
#                                     summary, name, scope)
#     x = fun(inputs, offsets)
#     return x


""" 2-D convolutional operation
"""
@core.layer
def conv2d(inputs, nouts,
           kshape=3,
           stride=1,
           padding=core.__defaults__['padding'],
           weight_initializer=core.__defaults__['weight_initializer'],
           weight_regularizer=core.__defaults__['weight_regularizer'],
           bias_initializer=core.__defaults__['bias_initializer'],
           bias_regularizer=core.__defaults__['bias_regularizer'],
           act=None,
           trainable=True,
           dtype=ops.core.float32,
           return_shape=False,
           collections=None,
           summary=core.__defaults__['summary'],
           reuse=False,
           name=None,
           scope=None):
    input_shape = ops.core.shape(inputs)
    fun, output = ops.convs.conv2d(input_shape,
                                   nouts,
                                   kshape,
                                   stride, padding,
                                   weight_initializer,
                                   weight_regularizer,
                                   bias_initializer,
                                   bias_regularizer,
                                   act,
                                   trainable,
                                   dtype,
                                   collections,
                                   summary,
                                   reuse,
                                   name,
                                   scope)
    return _layers(fun, inputs, output, return_shape,
                   'conv2d', reuse, name)


""" 2-D soft convolutional operation
    Returns
    =======
        x if return_offsets is False,
        x, offsets if return_offsets is true.
            mode = 'naive' / 'nearest'
                offsets in the form of [[batch, row, col, channel, kshape],
                                         ...
                                       ] for 2D, e.g., images
            mode = 'bilinear'
                offsets in the form of [[batch, row, col, channel, kernel_topleft],
                                        [batch, row, col, channel, kernel_topright],
                                        [batch, row, col, channel, kernel_bottomleft],
                                        [batch, row, col, channel, kernel_bottomright],
                                         ...
                                       ] for 2D, e.g., images
                where kernel_topleft, kernel_topright,
                      kernel_bottomleft, kernel_bottomright,
                are used to bilinear interpolation
            therefore, 'bilinear' have 4 times of length than 'naive' / 'nearest'
"""
@core.layer
def soft_conv2d(inputs, nouts,
                kshape=3,
                stride=1,
                padding=core.__defaults__['padding'],
                mode='bilinear',
                return_offsets=False,
                weight_initializer=core.__defaults__['weight_initializer'],
                weight_regularizer=core.__defaults__['weight_regularizer'],
                bias_initializer=core.__defaults__['bias_initializer'],
                bias_regularizer=core.__defaults__['bias_regularizer'],
                offset_weight_initializer='zeros',
                offset_weight_regularizer=None,
                offset_bias_initializer=None,
                offset_bias_regularizer=None,
                act=None,
                trainable=True,
                dtype=ops.core.float32,
                return_shape=False,
                collections=None,
                summary=core.__defaults__['summary'],
                reuse=False,
                name=None,
                scope=None):
    input_shape = ops.core.shape(inputs)
    fun, output = ops.convs.soft_conv2d(input_shape,
                                        nouts,
                                        kshape,
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
                                        act,
                                        trainable,
                                        dtype,
                                        collections,
                                        summary,
                                        reuse,
                                        name,
                                        scope)
    x, offsets = fun(inputs)
    xshape = ops.core.shape(x)
    if output != xshape:
        raise ValueError('the predicted output shape and the real'
                         ' output shape not match. {} vs {}'
                         .format(colors.green(output),
                                 colors.red(xshape)))
    if return_offsets:
        x = [x, offsets]
    if return_shape:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            x = [*x, output]
        else:
            x = [x, output]
    return x


""" 3-D convolutional operation
"""
@core.layer
def conv3d(inputs, nouts,
           kshape=3,
           stride=1,
           padding=core.__defaults__['padding'],
           weight_initializer=core.__defaults__['weight_initializer'],
           weight_regularizer=core.__defaults__['weight_regularizer'],
           bias_initializer=core.__defaults__['bias_initializer'],
           bias_regularizer=core.__defaults__['bias_regularizer'],
           act=None,
           trainable=True,
           dtype=ops.core.float32,
           return_shape=False,
           collections=None,
           summary=core.__defaults__['summary'],
           reuse=False,
           name=None,
           scope=None):
    # TODO: atruous_convxd
    input_shape = ops.core.shape(inputs)
    fun, output = ops.convs.conv3d(input_shape,
                                   nouts,
                                   kshape,
                                   stride,
                                   padding,
                                   weight_initializer,
                                   weight_regularizer,
                                   bias_initializer,
                                   bias_regularizer,
                                   act,
                                   trainable,
                                   dtype,
                                   collections,
                                   summary,
                                   reuse,
                                   name,
                                   scope)
    return _layers(fun, inputs, output, return_shape,
                   'conv3d', reuse, name)


""" 2-D transpose convolutional operation
    **TODO** atruos_convxd_tranpose
"""
@core.layer
def deconv2d(inputs, output_shape, nouts,
             kshape=3,
             stride=1,
             padding=core.__defaults__['padding'],
             weight_initializer=core.__defaults__['weight_initializer'],
             weight_regularizer=core.__defaults__['weight_regularizer'],
             bias_initializer=core.__defaults__['bias_initializer'],
             bias_regularizer=core.__defaults__['bias_regularizer'],
             act=None,
             trainable=True,
             dtype=ops.core.float32,
             return_shape=False,
             collections=None,
             summary=core.__defaults__['summary'],
             reuse=False,
             name=None,
             scope=None):
    input_shape = ops.core.shape(inputs)
    fun, output = ops.convs.deconv2d(input_shape,
                                     output_shape,
                                     nouts, kshape,
                                     stride, padding,
                                     weight_initializer,
                                     weight_regularizer,
                                     bias_initializer,
                                     bias_regularizer,
                                     act,
                                     trainable,
                                     dtype,
                                     collections,
                                     summary,
                                     reuse,
                                     name,
                                     scope)

    return _layers(fun, inputs, output, return_shape,
                   'deconv2d', reuse, name)


@core.layer
def sepconv2d(inputs, nouts,
              kshape=3,
              stride=1,
              padding=core.__defaults__['padding'],
              channel_multiplier=1,
              rate=1,
              weight_initializer=core.__defaults__['weight_initializer'],
              weight_regularizer=core.__defaults__['weight_regularizer'],
              bias_initializer=core.__defaults__['bias_initializer'],
              bias_regularizer=core.__defaults__['bias_regularizer'],
              act=None,
              trainable=True,
              dtype=ops.core.float32,
              return_shape=False,
              collections=None,
              summary=core.__defaults__['summary'],
              reuse=False,
              name=None,
              scope=None):
    input_shape = ops.core.shape(inputs)
    fun, output = ops.convs.sepconv2d(input_shape,
                                      nouts,
                                      kshape,
                                      stride,
                                      padding,
                                      channel_multiplier,
                                      rate,
                                      weight_initializer,
                                      weight_regularizer,
                                      bias_initializer,
                                      bias_regularizer,
                                      act, trainable,
                                      dtype,
                                      collections,
                                      summary,
                                      reuse,
                                      name,
                                      scope)
    return _layer(fun, inputs, output, return_shape,
                  'sepconv2d', reuse, name)
