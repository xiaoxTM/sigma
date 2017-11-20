from ..ops import pools
from .. import colors
from .layers import layers
import logging
import tensorflow as tf

@layers
def base_pool(inputs, op, psize, stride, padding, reuse=False, name=None):
    fun, output = op(inputs.get_shape().as_list(), psize, stride, padding, name)
    x = fun(inputs)
    helper.print_layer(inputs, x, op.__name__, reuse, name)
    if output != x.get_shape().as_list():
        raise ValueError('the predicted output shape and the real output shape not match. {} vs {}'
                         .format(output, x.get_shape().as_list()))
    return x

def avg_pool2d(inputs, psize, stride, padding, reuse=False, name=None):
    return base_pool(inputs, pools.avg_pool, psize, stride, padding, name)

def max_pool2d(inputs, psize, stride, padding, reuse=False, name=None):
    return base_pool(inputs, pools.max_pool, psize, stride, padding, name)
