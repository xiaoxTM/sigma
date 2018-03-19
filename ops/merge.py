import numpy as np
from .. import colors
from . import helper, core

def concat(inputs_shape,
           axis=-1,
           reuse=False,
           name=None,
           scope=None):
    """ concatenate multiple tensors into one
        along with axis
    """
    if not isinstance(inputs_shape, (list, tuple)):
        raise TypeError('concat requires inputs as '
                        '{}list / tpule{}, given {}{}{}'
                        .format(colors.fg.green, colors.reset, colors.fg.red,
                                type(inputs_shape), colors.reset))
    elif len(inputs_shape) < 2:
        raise ValueError('concat requires at least {}two{} inputs, given {}{}{}'
                         .format(colors.fg.green, colors.reset, colors.fg.red,
                                 len(inputs_shape), colors.reset))
    output_shape = inputs_shape[0]
    for idx, ip in enumerate(inputs_shape[1:]):
        if not np.all(np.delete(output_shape, axis) == np.delete(ip, axis)):
            raise ValueError('shape of {}{}{}-input differs from first'
                             ' one besides {}{}{}-axis. {}{} vs {}{}'
                             .format(colors.fg.red, idx+1, colors.reset,
                                     colors.fg.green, axis,
                                     colors.reset, colors.fg.red, output_shape,
                                     ip, colors.reset))
        output_shape[axis] += ip[axis]
    ops_scope, _, name = helper.assign_scope(name, scope, 'concat', reuse)
    def _concat(x):
        with ops_scope:
            return core.concat(x, axis, name)
    return _concat, output_shape


def add(inputs_shape,
        weights=None,
        reuse=False,
        name=None,
        scope=None):
    """ add multiple tensors into one
        with weight `weights`
    """
    if not isinstance(inputs_shape, (list, tuple)):
        raise TypeError('concat requires inputs as '
                        '{}list / tpule{}, given {}{}{}'
                        .format(colors.fg.green, colors.reset, colors.fg.red,
                                type(inputs_shape), colors.reset))
    elif len(inputs_shape) < 2:
        raise ValueError('concat requires at least {}two{} inputs, given {}{}{}'
                         .format(colors.fg.green, colors.reset, colors.fg.red,
                                 len(inputs_shape), colors.reset))
    output_shape = inputs_shape[0]
    for idx, ip in enumerate(inputs_shape[1:]):
        if not np.all(output_shape == ip):
            raise ValueError('shape of {}{}{}-input differ '
                             'from first one. {}{} vs {}{}'
                             .format(colors.fg.red, idx+1, colors.reset,
                                     colors.fg.red,
                                     output_shape, ip, colors.reset))
    if weights is None:
        weights = [1] * len(inputs_shape)
    ops_scope, _, name = helper.assign_scope(name, scope, 'add', reuse)
    def _add(x):
        with ops_scope:
            x = [e * w for e, w in zip(x, weights)]
            return core.add(x, name)
    return _add, output_shape


def mul(inputs_shape, reuse=False, name=None, scope=None):
    if not isinstance(inputs_shape, (list, tuple)):
        raise TypeError('concat requires inputs '
                        'as {}list / tpule{}, given {}{}{}'
                        .format(colors.fg.green, colors.reset, colors.fg.red,
                                type(inputs), colors.reset))
    elif len(inputs_shape) < 2:
        raise ValueError('concat requires at least {}two{} inputs, given {}{}{}'
                         .format(colors.fg.green, colors.reset, colors.fg.red,
                                 len(inputs_shape), colors.reset))
    output_shape = inputs_shape[0]
    for idx, ip in enumerate(inputs_shape[1:]):
        if not np.all(output_shape == ip):
            raise ValueError('shape of {}{}{}-input differ from '
                             'first one. {}{} vs {}{}'
                             .format(colors.fg.red, idx+1, colors.reset,
                                     colors.fg.red,
                                     output_shape, ip, colors.reset))
    ops_scope, _, name = helper.assign_scope(name, scope, 'mul', reuse)
    def _mul(x):
        with ops_scope:
            return core.multiply(inputs, name)
    return _mul
