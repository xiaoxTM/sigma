import tensorflow as tf

def base_pool2d(input_shape, fun, psize, stride, padding, name):
    if stride is None:
        stride = psize
    stride = helper.norm_input_2d(stride)
    psize = helper.norm_input_2d(psize)

    out_shape = helper.get_output_shape(input_shape, psize, stride, padding)

    if name is None:
        # if none and e.g., fun.__name__ == 'tf.nn.max_pool'
        #    name = max_pool
        name = fun.__name__.rsplit('.', 1)[1]
    scope = tf.name_scope(name)
    def _base_pool2d(x):
        with scope:
            return fun(x, psize, stride, padding.upper(), name)
    return _base_pool2d, out_shape


def avg_pool2d(input_shape, psize, stride=None, padding='same', name=None):
    return base_pool2d(input_shape, tf.nn.avg_pool, psize, stride, padding, name)

def max_pool2d(input_shape, psize, stride=None, padding='same', name=None):
    return base_pool2d(input_shape, tf.nn.max_pool, psize, stride, padding, name)
