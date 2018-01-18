from ..ops import pools, helper
from .. import colors
from .layers import layers
import logging
import tensorflow as tf

def _pool(inputs, op, psize, stride, padding,
          axis,
          return_shape,
          reuse,
          name,
          scope):
    fun, output = op(inputs.get_shape().as_list(), psize,
                     stride, padding, axis, reuse, name, scope)
    x = fun(inputs)
    # helper.print_layer(inputs, x, op.__name__, reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(output, x.get_shape().as_list()))
    if return_shape:
        x = [x, output]
    return x


def _pool_global(inputs,
                 op,
                 axis,
                 return_shape,
                 reuse,
                 name,
                 scope):
    fun, output = op(inputs.get_shape().as_list(), axis, reuse, name, scope)
    x = fun(inputs)
    # helper.print_layer(inputs, x, op.__name__, reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(output, x.get_shape().as_list()))
    if return_shape:
        x = [x, output]
    return x


@layers
def avg_pool2d(inputs, psize, stride, padding,
               axis=-1,
               return_shape=False,
               reuse=False,
               name='avg_pool2d',
               scope=None):
    return _pool(inputs, pools.avg_pool2d, psize,
                 stride, padding, axis, return_shape,
                 reuse, name, scope)


@layers
def avg_pool2d_global(inputs,
                      axis=-1,
                      return_shape=False,
                      reuse=False,
                      name='avg_pool2d_global',
                      scope=None):
    return _pool_global(inputs, pools.avg_pool2d_global,
                        axis, return_shape, reuse,
                        name, scope)


@layers
def max_pool2d(inputs, psize, stride, padding,
               axis=-1,
               return_shape=False,
               reuse=False,
               name='max_pool2d',
               scope=None):
    return _pool(inputs, pools.max_pool2d, psize,
                 stride, padding, axis, return_shape,
                 reuse, name, scope)


@layers
def max_pool2d_global(inputs,
                      axis=-1,
                      return_shape=False,
                      reuse=False,
                      name='max_pool2d_global',
                      scope=None):
    return _pool_global(inputs, pools.max_pool2d_global,
                        axis, return_shape, reuse,
                        name, scope)
