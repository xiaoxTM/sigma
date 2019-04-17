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
from .. import colors
from . import core

def _math(fun, inputs, output, typename, return_shape):
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if output[1:] != xshape[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(colors.green(output),
                                 colors.red(xshape)))
    if return_shape:
        x = [x, output]
    return x


@core.layer
def add(inputs,
        weights=None,
        return_shape=False,
        reuse=False,
        name=None,
        scope=None):
    input_shape = [ops.helper.norm_input_shape(ip)  for ip in inputs]
    fun, output = ops.math.add(input_shape, weights, reuse, name, scope)
    return _math(fun, inputs, output, 'add', return_shape)


@core.layer
def mul(inputs,
        return_shape=False,
        reuse=False,
        name=None,
        scope=None):
    input_shape = [ops.helper.norm_input_shape(ip) for ip in inputs]
    fun, output = ops.math.mul(input_shape, reuse, name, scope)
    return _math(fun, inputs, output, 'mul', return_shape)


@core.layer
def matmul(inputs,
           return_shape=False,
           reuse=False,
           name=None,
           scope=None):
    input_shape = [ops.helper.norm_input_shape(ip) for ip in inputs]
    fun, output = ops.math.matmul(input_shape, reuse, name, scope)
    return _math(fun, inputs, output, 'matmul', return_shape)
