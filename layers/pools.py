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

def _pool(inputs,
          op,
          pshape,
          stride,
          padding,
          return_shape,
          reuse,
          name,
          scope):
    fun, output = op(ops.core.shape(inputs), pshape,
                     stride, padding, reuse, name, scope)
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if output[1:] != xshape[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(output, xshape))
    if return_shape:
        x = [x, output]
    return x


def _pool_global(inputs,
                 op,
                 return_shape,
                 reuse,
                 name,
                 scope):
    fun, output = op(ops.helper.norm_input_shape(inputs), reuse, name, scope)
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if output[1:] != xshape[1:]:
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(output, xshape))
    if return_shape:
        x = [x, output]
    return x


@core.layer
def avg_pool2d(inputs,
               pshape=2,
               stride=None,
               padding=None,
               return_shape=False,
               reuse=False,
               name='avg_pool2d',
               scope=None):
    return _pool(inputs,
                 ops.pools.avg_pool2d,
                 pshape,
                 stride,
                 padding,
                 return_shape,
                 reuse,
                 name,
                 scope)


@core.layer
def avg_pool2d_global(inputs,
                      return_shape=False,
                      reuse=False,
                      name='avg_pool2d_global',
                      scope=None):
    return _pool_global(inputs,
                        ops.pools.avg_pool2d_global,
                        return_shape,
                        reuse,
                        name,
                        scope)


@core.layer
def max_pool2d(inputs,
               pshape=2,
               stride=None,
               padding='same',
               return_shape=False,
               reuse=False,
               name='max_pool2d',
               scope=None):
    return _pool(inputs,
                 ops.pools.max_pool2d,
                 pshape,
                 stride,
                 padding,
                 return_shape,
                 reuse,
                 name,
                 scope)


@core.layer
def max_pool2d_global(inputs,
                      return_shape=False,
                      reuse=False,
                      name='max_pool2d_global',
                      scope=None):
    return _pool_global(inputs,
                        ops.pools.max_pool2d_global,
                        return_shape,
                        reuse,
                        name,
                        scope)
