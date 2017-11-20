import tensorflow as tf
import numpy as np
from .. import colors

def concat(inputs_shape, axis=-1, name=None):
    if not isinstance(inputs_shape, (list, tuple)):
        raise TypeError('concat requires inputs as {}list / tpule{}, given {}{}{}'
                        .format(colors.fg.green, colors.reset, colors.fg.red,
                                type(inputs_shape), colors.reset))
    elif len(inputs_shape) < 2:
        raise ValueError('concat requires at least {}two{} inputs, given {}{}{}'
                         .format(colors.fg.green, colors.reset, colors.fg.red,
                                 len(inputs_shape), colors.reset))
    output_shape = inputs_shape[0]
    for idx, ip in enumerate(inputs_shape[1:]):
        if not np.all(np.delete(output_shape, axis) == np.delete(ip, axis)):
            raise ValueError('shape of {}{}{}-input differ from first one besides {}{}{}-axis. {}{} vs {}{}'
                             .format(colors.fg.red, idx, colors.reset, colors.fg.green, axis,
                                     colors.reset, colors.fg.red, output_shape, ip,
                                     colors.reset))
        output_shape[axis] += ip[axis]
    if name is None:
        name = helper.dispatch_name('concat')
    scope = tf.name_scope(name)
    def _concat(x):
        with scope:
            return tf.concat(x, axis, name)
    return _concat, output_shape

def add(inputs_shape, name=None):
    if not isinstance(inputs_shape, (list, tuple)):
        raise TypeError('concat requires inputs as {}list / tpule{}, given {}{}{}'
                        .format(colors.fg.green, colors.reset, colors.fg.red,
                                type(inputs_shape), colors.reset))
    elif len(inputs_shape) < 2:
        raise ValueError('concat requires at least {}two{} inputs, given {}{}{}'
                         .format(colors.fg.green, colors.reset, colors.fg.red,
                                 len(inputs_shape), colors.reset))
    output_shape = inputs_shape[0]
    for ip in inputs_shape[1:]:
        if not np.all(output_shape != ip):
            raise ValueError('shape of {}{}{}-input differ from first one. {}{} vs {}{}'
                             .format(color.fg.red, colors.reset, colors.fg.red,
                                     output_shape, ip, colors.reset))
    if name is None:
        name = helper.dispatch_name('add')
    scope = tf.name_scope(name)
    def _add(x):
        with scope:
            return tf.add_n(x, name)
    return _add, output_shape

def mul(inputs, name=None):
    if not isinstance(inputs, (list, tuple)):
        raise TypeError('concat requires inputs as {}list / tpule{}, given {}{}{}'
                        .format(colors.fg.green, colors.reset, colors.fg.red,
                                type(inputs), colors.reset))
    elif len(inputs) != 2:
        raise ValueError('concat requires only {}two{} inputs, given {}{}{}'
                         .format(colors.fg.green, colors.reset, colors.fg.red,
                                 len(inputs), colors.reset))
    output_shape = inputs[0]
    for ip in inputs[1:]:
        if not np.all(output_shape != ip):
            raise ValueError('shape of {}{}{}-input differ from first one. {}{} vs {}{}'
                             .format(color.fg.red, colors.reset, colors.fg.red,
                                     output_shape, ip, colors.reset))
    if name is None:
        name = helper.dispatch_name('mul')
    scope = tf.name_scope(name)
    def _mul(x):
        with scope:
            return tf.multiply(inputs, name)

    return _mul
