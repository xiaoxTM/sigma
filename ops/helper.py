import tensorflow as tf
from .. import colors, status
import numpy as np
import copy

def name_space():
    name_maps = {}
    def _name_space(x):
        if not isinstance(x, str):
            raise TypeError('name space requires string but given {}'
                           .format(type(x)))
        if x not in name_maps.keys():
            name_maps[x] = 0
        nid = name_maps[x]
        name_maps[x] += 1
        return '{}-{}'.format(x, nid)
    return _name_space

dispatch_name = name_space()

def is_tensor(x):
    return tf.contrib.framework.is_tensor(x)


""" if not reuse:
        print('{}{}{} \t\t=>`[{} | {}]`=> \t\t{}{}{}'
              .format(colors.fg.green, input_shape, colors.reset,
                      name if name is not None else 'concat', 'concat',
                      colors.fg.red, output, colors.reset))
"""
def print_layer(inputs, outputs, typename, reuse=False, name=None):
    if not reuse:
        if isinstance(inputs, (list, tuple)):
            input_shape = [ip.get_shape().as_list() for ip in inputs]
            inputname = 'merge'
        else:
            input_shape = inputs.get_shape().as_list()
            inputname = inputs.name
        output_shape = outputs.get_shape().as_list()
        print('{}{}{}{} \t\t=>`{}[{} | {}]{}`=> \t\t{}{}{}{}'
              .format(inputname, colors.fg.green, input_shape, colors.reset,
                      colors.fg.blue,
                      name if name is not None else typename,
                      typename, colors.reset, outputs.name,
                      colors.fg.red, output_shape, colors.reset))

""" normalize axis given tensor shape
    for example: tensor shape [batch size, rows, cols, channels], axis = -1
        return axis=3
"""
def normalize_axes(tensor_shape, axis=status.axis):
    if not isinstance(tensor_shape, (list, tuple)):
        raise TypeError('tensor shape must be list/tuple, given {}{}{}[{}]'
                        .format(colors.fg.red,
                                type(tensor_shape),
                                colors.reset,
                                tensor_shape))
    input_len = len(tensor_shape)
    if isinstance(axis, int):
        axis = (axis + input_len) % input_len
    elif isinstance(axis, (list, tuple)):
        axis = map(lambda x:(x + input_len) % input_len, axis)
    else:
        raise TypeError('axis must be int or list/tuple, given {}{}{}[{}]'
                        .format(colors.fg.red,
                                type(axis),
                                colors.reset,
                                axis))
    return axis


def get_output_shape(input_shape, nouts, kernel, stride, padding):
    typecheck = map(lambda x, y: isinstance(x, y),
                    [input_shape, kernel, stride],
                    [(list, tuple)]*3)
    if not np.all(typecheck):
        raise TypeError('type of input, kernel, stride not all '
                        'list / tuple, given{}{}{}, {}{}{}, {}{}{}'
                        .format(colors.fg.red, type(input_shape), colors.reset,
                                colors.fg.red, type(kernel), colors.reset,
                                colors.fg.red, type(stride), colors.reset))

    if len(input_shape) != len(kernel) or \
       len(input_shape) != len(stride):
        raise ValueError('shape of input, kernel, stride not match,'
                         ' given {}{}{}, {}{}{}, {}{}{}'
                         .format(colors.fg.red, len(input_shape), colors.reset,
                                 colors.fg.red, len(kernel), colors.reset,
                                 colors.fg.red, len(stride), colors.reset))

    padding = padding.upper()
    if padding not in ['VALID', 'SAME']:
        raise ValueError('padding be either {}VALID{} or {}SAME{}, given {}{}{}'
                         .format(colors.fg.green, colors.reset,
                                 colors.fg.green ,colors.reset,
                                 colors.fg.red, padding, colors.reset))
    out_shape = copy.deepcopy(input_shape)
    index = range(len(input_shape))
    if padding == 'SAME':
        for idx in index[1:-1]:
            out_shape[idx] = int(
                np.ceil(float(input_shape[idx]) / float(stride[idx])))
    else:
        for idx in index[1:-1]:
            # NOTE: unlike normal convolutional operation, which is:
            #           ceil((image-size - kernel-size) / stride) + 1
            #       tensorflow calculate the output shape in another way:
            #           ceil((image-size - kernel-size + 1) / stride)
            out_shape[idx] = int(
              np.ceil(
                float(input_shape[idx] - kernel[idx] + 1) / float(stride[idx])
            ))
    out_shape[-1] = nouts
    return out_shape

def norm_input_1d(shape):
    if isinstance(shape, int):
        shape = [1, shape, 1]
    elif isinstance(shape, (list, tuple)):
        if len(shape) == 1:
            shape = [1, shape[0], 1]
        elif len(shape) != 3:
            raise ValueError('conv1d require input shape {}[batch-size,'
                             ' cols, channels]{}, given {}{}{}'
                             .format(colors.fg.green, colors.reset,
                                     colors.fg.red, input_shape, colors.reset))
    else:
        raise TypeError('kernel for conv1d require {}int/list/tuple{} '
                        'type, given {}`{}`{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.red, type(kernel), colors.reset))
    return shape

def norm_input_2d(shape):
    if isinstance(shape, int):
        shape = [1, shape, shape, 1]
    elif isinstance(shape, (list, tuple)):
        if len(shape) == 1:
            shape = [1, shape[0], shape[0], 1]
        elif len(shape) == 2:
            shape = [1, shape[0], shape[1], 1]
        elif len(shape) != 4:
            raise ValueError('conv1d require input shape {}[batch-size, '
                             'rows, cols, channels]{}, given {}{}{}'
                             .format(colors.fg.green, colors.reset,
                                     colors.fg.red, input_shape, colors.reset))
    else:
        raise TypeError('kernel for conv1d require '
                        '{}int/list/tuple{} type, given {}`{}`{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.red, type(kernel), colors.reset))
    return shape

def norm_input_3d(shape):
    if isinstance(shape, int):
        shape = [1, shape, shape, shape, 1]
    elif isinstance(shape, (list, tuple)):
        if len(shape) == 1:
            shape = [1, shape[0], shape[0], shape[0], 1]
        elif len(shape) == 3:
            shape = [1, shape[0], shape[1], shape[2], 1]
        elif len(shape) != 5:
            raise ValueError('conv1d require input shape {}[batch-size, '
                             'depths, rows, cols, channels]{}, given {}{}{}'
                             .format(colors.fg.green, colors.reset,
                                     colors.fg.red, input_shape, colors.reset))
    else:
        raise TypeError('kernel for conv1d require '
                        '{}int/list/tuple{} type, given {}`{}`{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.red, type(kernel), colors.reset))
