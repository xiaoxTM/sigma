import tensorflow as tf
from .. import colors
from . import core
import numpy as np
import copy
import collections
import os.path


def name_space():
    name_maps = collections.OrderedDict()
    def _name_space(x=None, index=None):
        if x is None and index is None:
            raise TypeError('`x` and `index` cannot both be None')
        if x is not None and not isinstance(x, str):
            raise TypeError('name space requires string but given {}'
                           .format(type(x)))
        if index is not None and not isinstance(index, int):
            raise TypeError('index space requires int but given {}'
                           .format(type(index)))
        if x is None:
            return '{}-{}'.format(*list(name_maps.items())[index])
        else:
            if index is None:
                if x not in name_maps.keys():
                    name_maps[x] = -1
                nid = name_maps[x] + 1
                name_maps[x] = nid
                return '{}-{}'.format(x, nid)
            else:
                if x not in name_maps.keys():
                    raise ValueError('`{}` not in name maps for indexing {}'
                                     .format(x, index))
                size = name_maps[x] + 1
                nid = (size + index) % size
                return '{}-{}'.format(x, nid)
    return _name_space


dispatch_name = name_space()


# assign_scope will return the operation scope
#    and append layer-type to name
# for example, given name = 'block' and ltype = conv
#     assign_scope return layer = block/conv
def assign_scope(name, scope, ltype, reuse=False):
    if name is None and ltype is None:
        raise ValueError('Either `name` or `ltype` must be None')
    if name is None:
        if reuse:
            name = dispatch_name(ltype, -1)
        else:
            name = dispatch_name(ltype)
    name_with_ltype = '{}/{}'.format(name, ltype)
    if scope is None:
        ops_scope = tf.name_scope('{}'.format(name_with_ltype))
    else:
        ops_scope = tf.name_scope('{}/{}'
                                  .format(scope, name_with_ltype))
    return ops_scope, name_with_ltype, name


def is_tensor(x):
    return tf.contrib.framework.is_tensor(x)


def depth(x):
    return core.shape(x)[core.axis]


def feature_dims(x):
    dims = list(range(core.rank(x)))
    shape.pop(core.axis)
    shape.pop(0)
    return shape


def name_normalize(names):
    """ normalize variable name
        generally, `names` is a list of :
            [scope/]/{layer-name/layer-type}/variable-name:index
    """
    def _normalize(name):
        splits = name.rsplit('/', 2)
        if len(splits) == 1:
            return splits[0].split(':')[0]
        elif len(splits) == 2:
            return name.split(':')[0]
        else:
            return splits[0]
    if isinstance(names, str):
        return _normalize(names)
    elif isinstance(names, (tuple, list, np.ndarray)):
        return map(_normalize, names)
    else:
        raise TypeError('name must be tuple/list/np.ndarray. given {}'
                        .format(type(names)))


""" normalize axis given tensor shape
    for example: tensor shape [batch size, rows, cols, channels], axis = -1
        return axis=3
"""
def normalize_axes(tensor_shape, axis=core.axis):
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


def get_output_shape(input_shape, nouts, kshape, stride, padding):
    typecheck = map(lambda x, y: isinstance(x, y),
                    [input_shape, kshape, stride],
                    [(list, tuple)]*3)
    if not np.all(typecheck):
        raise TypeError('type of input, kshape, stride not all '
                        'list / tuple, given{}{}{}, {}{}{}, {}{}{}'
                        .format(colors.fg.red, type(input_shape), colors.reset,
                                colors.fg.red, type(kshape), colors.reset,
                                colors.fg.red, type(stride), colors.reset))

    if len(input_shape) != len(kshape) or \
       len(input_shape) != len(stride):
        raise ValueError('shape of input, kshape, stride not match,'
                         ' given {}{}{}, {}{}{}, {}{}{}'
                         .format(colors.fg.red, len(input_shape), colors.reset,
                                 colors.fg.red, len(kshape), colors.reset,
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
            #           ceil((image-size - kshape-size) / stride) + 1
            #       tensorflow calculate the output shape in another way:
            #           ceil((image-size - kshape-size + 1) / stride)
            out_shape[idx] = int(
              np.ceil(
                float(input_shape[idx] - kshape[idx] + 1) / float(stride[idx])
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
            raise ValueError('require input shape {}[batch-size,'
                             ' cols, channels]{}, given {}{}{}'
                             .format(colors.fg.green, colors.reset,
                                     colors.fg.red, shape, colors.reset))
    else:
        raise TypeError('shape require {}int/list/tuple{} type, given {}`{}`{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.red, type(shape), colors.reset))
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
            raise ValueError('require input shape {}[batch-size, '
                             'rows, cols, channels]{}, given {}{}{}'
                             .format(colors.fg.green, colors.reset,
                                     colors.fg.red, shape, colors.reset))
    else:
        raise TypeError('shape requires {}int/list/tuple{} type, given {}`{}`{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.red, type(shape), colors.reset))
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
            raise ValueError('require input shape {}[batch-size, '
                             'depths, rows, cols, channels]{}, given {}{}{}'
                             .format(colors.fg.green, colors.reset,
                                     colors.fg.red, input_shape, colors.reset))
    else:
        raise TypeError('shape requires {}int/list/tuple{} type, given {}`{}`{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.red, type(kshape), colors.reset))
