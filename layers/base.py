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
from .. import ops
from . import core

@core.layer
def embedding(inputs,
              table_size,
              strategy='mod',
              dtype=ops.core.float32,
              initializer='glorot_uniform',
              regularizer=None,
              cpuid=0,
              trainable=True,
              collections=None,
              summary='histogram',
              reuse=False,
              name=None,
              scope=None):
    fun = ops.convs.embedding(table_size,
                              strategy,
                              dtype,
                              initializer,
                              regularizer,
                              cpuid,
                              trainable,
                              collections,
                              summary,
                              reuse,
                              name,
                              scope)
    x = fun(inputs)
    return x


@core.layer
def flatten(inputs,
            return_shape=False,
            reuse=False,
            name=None,
            scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.base.flatten(input_shape, reuse, name, scope)
    x = fun(inputs)
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
def reshape(inputs,
            output_shape,
            return_shape=False,
            reuse=False,
            name=None,
            scope=None):
    # input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.base.reshape(output_shape, reuse, name, scope)
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if output[1:] != output_shape[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(colors.red(output),
                                 colors.green(xshape)))
    if return_shape:
        x = [x, output]
    return x


@core.layer
def transpose(inputs,
              perm,
              conjugate=False,
              return_shape=False,
              reuse=False,
              name=None,
              scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.base.transpose(input_shape,
                                     perm,
                                     conjugate,
                                     reuse,
                                     name,
                                     scope)
    x = fun(inputs)
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
def expand_dims(inputs,
                axis,
                return_shape=False,
                reuse=False,
                name=None,
                scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    fun, output = ops.base.expand_dims(input_shape, axis, reuse, name, scope)
    x = fun(inputs)
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
def maskout(inputs,
            index=None,
            axis=-2, # axis according to which to maskout
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
    fun, output = ops.base.maskout(input_shape,
                                   axis,
                                   drop,
                                   flatten,
                                   reuse,
                                   name,
                                   scope)
    x = fun(inputs, index)
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
def input_spec(inputs,
               dtype=ops.core.float32,
               reuse=False,
               name=None,
               scope=None):
    """ inputs is a list / tuple in logic
        due to core.layer spec, the first
        parameter must be `inputs`.
        therefore use `inputs` instead of `input_shape`
        for naming parameter
    """
    ops.helper.check_input_shape(inputs)
    ops_scope, name_with_ltype, name = ops.helper.assign_scope(name,
                                                               scope,
                                                               'inputs',
                                                               reuse)
    return ops.core.placeholder(dtype, inputs, name)


@core.layer
def label_spec(inputs,
               dtype=ops.core.int32,
               reuse=False,
               name=None,
               scope=None):
    """ inputs is a list / tuple in logic
        due to core.layer spec, the first
        parameter must be `inputs`.
        therefore use `inputs` instead of `input_shape`
    """
    ops.helper.check_input_shape(inputs)
    ops_scope, name_with_ltype, name = ops.helper.assign_scope(name,
                                                               scope,
                                                               'labels',
                                                               reuse)
    return ops.core.placeholder(dtype, inputs, name)


@core.layer
def random_spec(inputs,
                dtype=ops.core.float32,
                type='random_uniform',
                reuse=False,
                name=None,
                scope=None,
                **kwargs):
    ops.helper.check_input_shape(inputs)
    ops_scope, name_with_ltype, name = ops.helper.assign_scope(name,
                                                               scope,
                                                               'random',
                                                               reuse)
    kwargs['name'] = name
    initializer = ops.initializers.get(type, **kwargs)
    return initializer(inputs, dtype)
