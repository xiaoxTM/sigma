from ..ops import pools, helper
from .. import colors
from .layers import layers
import logging
import tensorflow as tf

def base_pool(inputs, op, psize, stride, padding,
              axis=-1, return_shape=False, reuse=False,
              name=None, scope=None):
    fun, output = op(inputs.get_shape().as_list(), psize,
                     stride, padding, axis, name, scope)
    x = fun(inputs)
    helper.print_layer(inputs, x, op.__name__, reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(output, x.get_shape().as_list()))
    if return_shape:
        x = [x, output]
    return x

def base_pool_global(inputs, op, axis=-1, return_shape=False,
                     reuse=False, name=None, scope=None):
    fun, output = op(inputs.get_shape().as_list(), axis, reuse, name, scope)
    x = fun(inputs)
    helper.print_layer(inputs, x, op.__name__, reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the '
                         'real output shape not match. {} vs {}'
                         .format(output, x.get_shape().as_list()))
    if return_shape:
        x = [x, output]
    return x

@layers
def avg_pool2d(inputs, psize, stride, padding,
               axis=-1, return_shape=False,
               reuse=False, name=None, scope=None):
    return base_pool(inputs, pools.avg_pool2d, psize,
                     stride, padding, axis, return_shape, name, scope)

@layers
def avg_pool2d_global(inputs, axis=-1, return_shape=False,
                      reuse=False, name=None, scope=None):
    return base_pool_global(inputs, pools.avg_pool2d_global,
                            axis, return_shape, reuse, name, scope)

@layers
def max_pool2d(inputs, psize, stride, padding,
               axis=-1,  return_shape=False, reuse=False,
               name=None, scope=None):
    return base_pool(inputs, pools.max_pool2d, psize,
                     stride, padding, axis, return_shape, name, scope)

@layers
def max_pool2d_global(inputs, axis=-1, return_shape=False,
                      reuse=False, name=None, scope=None):
    return base_pool_global(inputs, pools.max_pool2d_global,
                            axis, return_shape, reuse, name, scope)
