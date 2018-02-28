from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

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
    if output != xshape:
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
    fun, output = op(ops.core.shape(inputs), reuse, name, scope)
    x = fun(inputs)
    xshape = ops.core.shape(x)
    if output != xshape:
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
                 pools.avg_pool2d,
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
                        pools.avg_pool2d_global,
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
                 pools.max_pool2d,
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
                        pools.max_pool2d_global,
                        return_shape,
                        reuse,
                        name,
                        scope)
