from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from . import helper

def base_pool2d(input_shape, fun, pshape, stride,
                padding, reuse, name, scope):
    if stride is None:
        stride = pshape
    stride = helper.norm_input_2d(stride)
    pshape = helper.norm_input_2d(pshape)
    out_shape = helper.get_output_shape(input_shape, input_shape[core.axis],
                                        pshape, stride, padding)

    ltype = fun.__name__.rsplit('.', 1)
    if len(ltype) > 1:
        ltype = ltype[1]
    else:
        ltype = ltype[0]
    ops_scope, name = helper.assign_scope(name, scope, ltype, reuse)
    def _base_pool2d(x):
        with ops_scope:
            return fun(x, pshape, stride, padding.upper(),
                       data_format=core.data_format, name=name)
    return _base_pool2d, out_shape


def base_pool2d_global(input_shape, fun, reuse, name, scope):
    # if none and e.g., fun.__name__ == 'tf.nn.max_pool'
    #    name = max_pool
    ltype = fun.__name__.rsplit('.', 1)
    if len(ltype) > 1:
        ltype = ltype[1]
    else:
        ltype = ltype[0]
    ops_scope, name = helper.assign_scope(name, scope, ltype, reuse)
    axes = [idx for idx, _ in enumerate(input_shape)]
    del axes[core.axis]
    del axes[0]
    def _base_pool2d_global(x):
        with ops_scope:
            return fun(x, axis=axes, name=name)
    return _base_pool2d_global, [input_shape[0], input_shape[core.axis]]


def avg_pool2d(input_shape,
               pshape=2,
               stride=None,
               padding='same',
               reuse=False,
               name=None,
               scope=None):
    return base_pool2d(input_shape, tf.nn.avg_pool, pshape,
                       stride, padding, reuse, name, scope)


def avg_pool2d_global(input_shape,
                      reuse=False,
                      name=None,
                      scope=None):
    return base_pool2d_global(input_shape, tf.reduce_mean,
                              reuse, name, scope)


def max_pool2d(input_shape,
               pshape=2,
               stride=None,
               padding='same',
               reuse=False,
               name=None,
               scope=None):
    return base_pool2d(input_shape, tf.nn.max_pool, pshape,
                       stride, padding, reuse, name, scope)


def max_pool2d_global(inputs,
                      reuse=False,
                      name=None,
                      scope=None):
    return base_pool2d_global(input_shape, tf.reduce_max,
                              reuse, name, scope)


def resize(input_shape,
           output_shape=None,
           factor=None,
           mode='bilinear',
           align_corners=False,
           reuse=False,
           name=None,
           scope=None):
    if output_shape is None:
        if factor is None:
            raise ValueError('cannot feed bilinear with both '
                             'none of output_shape and factor')
        output_shape = [x for x in input_shape]
        if isinstance(factor, (int, float)):
            factor = [1] + [factor] * (len(output_shape)-2) + [1]
        elif isinstance(factor, (list, tuple)):
            if len(factor) == 1:
                factor = [1] + factor * (len(output_shape) - 2) + [1]
            elif len(factor) == (len(output_shape) - 2):
                factor = [1] + factor + [1]
            elif len(factor) != len(output_shape):
                raise ValueError('factor and output_shape '
                                 'length not match. {} vs {}'
                                 .format(factor, output_shape))
        else:
            raise TypeError('factor type not support in bilinear')
        for i in range(1, len(output_shape)-1):
            output_shape[i] *= factor[i]
    if mode not in ['bilinear', 'bicubic', 'area', 'nearest_neighbor']:
        raise ValueError('mode must be one of '
                         '[bilinear, bicubic, area, nearest_neighbor]')
    ops_scope, name = helper.assign_scope(name, scope, model, reuse)
    def _resize(x):
        with ops_scope:
            return eval('tf.image.resize_{}(x, output_shape, align_corners, name)'
                        .format(mode))
    return _resize
