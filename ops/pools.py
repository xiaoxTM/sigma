import tensorflow as tf

from . import helper

def base_pool2d(input_shape, fun, psize, stride, padding, axis, name, scope):
    if stride is None:
        stride = psize
    stride = helper.norm_input_2d(stride)
    psize = helper.norm_input_2d(psize)

    out_shape = helper.get_output_shape(input_shape, input_shape[axis],
                                        psize, stride, padding)

    if name is None:
        # if none and e.g., fun.__name__ == 'tf.nn.max_pool'
        #    name = max_pool
        name = fun.__name__.rsplit('.', 1)
        if len(name) > 1:
            name = name[1]
        else:
            name = name[0]
    if scope is None:
        scope = name
    scope = tf.name_scope(scope)
    def _base_pool2d(x):
        with scope:
            return fun(x, psize, stride, padding.upper(), name=name)
    return _base_pool2d, out_shape

def base_pool2d_global(input_shape, fun, axis, reuse, name, scope):
    if name is None:
        # if none and e.g., fun.__name__ == 'tf.nn.max_pool'
        #    name = max_pool
        name = fun.__name__.rsplit('.', 1)
        if len(name) > 1:
            name = name[1]
        else:
            name = name[0]
    if scope is None:
        scope = name
    scope = tf.name_scope(scope)

    axes = [idx for idx, _ in enumerate(input_shape)]
    del axes[axis]
    del axes[0]

    def _base_pool2d_global(x):
        with scope:
            return fun(x, axis=axes, name=name)
    return _base_pool2d_global, [input_shape[0], input_shape[axis]]


def avg_pool2d(input_shape, psize, stride=None,
               padding='same', axis=-1, name=None, scope=None):
    return base_pool2d(input_shape, tf.nn.avg_pool, psize,
                       stride, padding, axis, name, scope)


def avg_pool2d_global(input_shape, axis=-1, reuse=False,
                      name=None, scope=None):
    return base_pool2d_global(input_shape, tf.reduce_mean,
                              axis, reuse, name, scope)


def max_pool2d(input_shape, psize, stride=None,
               padding='same', axis=-1, name=None, scope=None):
    return base_pool2d(input_shape, tf.nn.max_pool, psize,
                       stride, padding, axis, name, scope)


def max_pool2d_global(inputs, axis=-1, reuse=False, name=None, scope=None):
    return base_pool2d_global(input_shape, tf.reduce_max,
                              axis, reuse, name, scope)


def resize(input_shape, output_shape, factor=None, mode='bilinear',
           align_corners=False, reuse=False, name=None, scope=None):
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

    if name is None:
        # if none and e.g., fun.__name__ == 'tf.nn.max_pool'
        #    name = max_pool
        name = fun.__name__.rsplit('.', 1)
        if len(name) > 1:
            name = name[1]
        else:
            name = name[0]
    if scope is None:
        scope = name
    scope = tf.name_scope(scope)

    def _resize(x):
        with scope:
            return eval('tf.image.resize_{}(x, output_shape, align_corners, name)'
                        .format(mode))
    return _resize
