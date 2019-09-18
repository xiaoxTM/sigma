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

from .. import ops
from . import core

@core.layer
def instance_norm(inputs,
                  offset_initializer='zeros',
                  scale_initializer='ones',
                  offset_regularizer=None,
                  scale_regularizer=None,
                  cpuid=0,
                  epsilon=ops.core.epsilon,
                  act=None,
                  trainable=True,
                  collections=None,
                  summary='histogram',
                  reuse=False,
                  name=None,
                  scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun = ops.norms.instance_norm(input_shape,
                                  offset_initializer,
                                  scale_initializer,
                                  offset_regularizer,
                                  scale_regularizer,
                                  cpuid,
                                  epsilon,
                                  act,
                                  trainable,
                                  collections,
                                  summary,
                                  check_input_shape,
                                  reuse,
                                  name,
                                  scope)
    x = core.run_and_record_fun(fun, name, inputs)
    if check_output_shape:
        xshape = ops.core.shape(x)
        if input_shape[1:] != xshape[1:]:
            raise ValueError('the predicted output shape and the '
                             'real output shape not match. {} vs {}'
                             .format(colors.green(input_shape),
                                     colors.red(xshape)))
    return x


@core.layer
def conditional_instance_norm(inputs,
                              bank_size,
                              offset_initializer='zeros',
                              scale_initializer='ones',
                              offset_regularizer=None,
                              scale_regularizer=None,
                              cpuid=0,
                              epsilon=ops.core.epsilon,
                              act=None,
                              trainable=True,
                              collections=None,
                              summary='histogram',
                              reuse=False,
                              name=None,
                              scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun = ops.norms.conditional_instance_norm(input_shape,
                                              bank_size,
                                              offset_initializer,
                                              scale_initializer,
                                              offset_regularizer,
                                              scale_regularizer,
                                              cpuid,
                                              epsilon,
                                              act,
                                              trainable,
                                              collections,
                                              summary,
                                              check_input_shape,
                                              reuse,
                                              name,
                                              scope)
    x = core.run_and_record_fun(fun, name, inputs)
    if check_output_shape:
        xshape = ops.core.shape(x)
        if input_shape[1:] != xshape[1:]:
            raise ValueError('the predicted output shape and the '
                             'real output shape not match. {} vs {}'
                             .format(colors.green(input_shape),
                                     colors.red(xshape)))
    return x


@core.layer
def batch_norm(inputs,
               axes=None,
               momentum=0.99,
               offset_initializer='zeros',
               scale_initializer='ones',
               offset_regularizer=None,
               scale_regularizer=None,
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               cpuid=0,
               epsilon=ops.core.epsilon,
               act=None,
               trainable=True,
               fused=True,
               collections=None,
               summary='histogram',
               reuse=False,
               name=None,
               scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun = ops.norms.batch_norm(input_shape,
                               axes,
                               momentum,
                               offset_initializer,
                               scale_initializer,
                               offset_regularizer,
                               scale_regularizer,
                               moving_mean_initializer,
                               moving_variance_initializer,
                               cpuid,
                               epsilon,
                               act,
                               trainable,
                               collections,
                               summary,
                               check_input_shape,
                               reuse,
                               name,
                               scope)
    x = core.run_and_record_fun(fun, name, inputs, is_training, fused)
    if check_output_shape:
        xshape = ops.core.shape(x)
        if input_shape[1:] != xshape[1:]:
            raise ValueError('the predicted output shape and the '
                             'real output shape not match. {} vs {}'
                             .format(colors.green(input_shape),
                                     colors.red(xshape)))
    return x
