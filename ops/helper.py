"""
    sigma, a deep neural network framework.
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from .. import colors
from . import core
import numpy as np
import copy
import collections
import os.path
from contextlib import contextmanager

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
        ops_scope = core.name_scope('{}'.format(name_with_ltype))
    else:
        ops_scope = core.name_scope('{}/{}'
                                  .format(scope, name_with_ltype))
    return ops_scope, name_with_ltype, name


@contextmanager
def maybe_layer(aslayer=False, name=None, scope=None, ltype=None, reuse=False):
    if aslayer:
        ops_scope = assign_scope(name, scope, ltype, reuse)
        yield ops_scope
    else:
        yield


def is_tensor(x):
    return core.is_tensor(x)


def depth(x):
    return core.shape(x)[core.axis]


def feature_dims(x):
    dims = list(range(core.rank(x)))
    shape.pop(core.axis)
    shape.pop(0)
    return shape


def name_normalize(names, scope=None):
    """ normalize variable name (or say, remove variable index)
        generally, `names` is a list of :
            [scope/]/{layer-name/layer-type}/[{sub-spaces/}*]variable-name:index
    """
    def _normalize(name):
        name = name.rsplit(':', 1)[0]
        if scope is None:
            return name.split('/', 1)[0]
        else:
            return name.splie('/', 2)[1]
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
        raise TypeError('tensor shape must be list/tuple, given {}[{}]'
                        .format(colors.red(type(tensor_shape)),
                                tensor_shape))
    input_len = len(tensor_shape)
    if isinstance(axis, int):
        axis = (axis + input_len) % input_len
    elif isinstance(axis, (list, tuple)):
        axis = map(lambda x:(x + input_len) % input_len, axis)
    else:
        raise TypeError('axis must be int or list/tuple, given {}[{}]'
                        .format(colors.red(type(axis)),
                                axis))
    return axis


def get_output_shape(input_shape, nouts, kshape, stride, padding):
    """ get the corresponding output shape given tensor shape
        Attributes
        ==========
            input_shape : list / tuple
                          input tensor shape
            nouts : int
                    number of output feature maps
            kshape : int / list / tuple
                     kernel shape for convolving operation
            stride : int / list / tuple
                     stride for convolving operation
            padding : string
                      padding for convolving operation
        Returns
        ==========
            output tensor shape of convolving operation
    """
    typecheck = map(lambda x, y: isinstance(x, y),
                    [input_shape, kshape, stride],
                    [(list, tuple)]*3)
    if not np.all(typecheck):
        raise TypeError('type of input, kshape, stride not all '
                        'list / tuple, given {}, {}, {}'
                        .format(colors.red(type(input_shape)),
                                colors.red(type(kshape)),
                                colors.red(type(stride))))

    if len(input_shape) != len(kshape) or \
       len(input_shape) != len(stride):
        raise ValueError('shape of input, kshape, stride not match,'
                         ' given {}, {}, {}'
                         .format(colors.red(len(input_shape)),
                                 colors.red(len(kshape)),
                                 colors.red(len(stride))))

    padding = padding.upper()
    if padding not in ['VALID', 'SAME']:
        raise ValueError('padding be either {}VALID{} or {}SAME{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.fg.green ,colors.reset,
                                 colors.red(padding)))
    out_shape = input_shape[:]
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


def norm_input_shape(input_tensor):
    """ get the input shape
        if the first axis is None,
        will be assigned to Tensor as batch_size
    """
    input_shape = core.shape(input_tensor)
    if input_shape[0] is None:
        input_shape[0] = core.tshape(input_tensor)[0]
    return input_shape


def check_input_shape(input_shape):
    """ check input shape
        input shape can have one -1 or one None
        or scalar tensor in batch-size axis
    """
    stat = core.shape_statistics(input_shape)
    if len(stat['nones']) > 1:
        raise ValueError('input shape `{}` contains more than one None'
                         .format(colors.red(input_shape)))
    elif len(stat['-1']) > 1:
        raise ValueError('input shape `{}` contains more than one -1'
                         .format(colors.red(input_shape)))
    elif len(stat['nones']) == 1 and len(stat['-1']) == 1:
        raise ValueError('input shape `{}` contains both None and -1'
                         .format(colors.red(input_shape)))


def norm_input_1d(shape):
    """ norm input for 1d convolving operation
        generally called by norm kernel-shape and stride
        e.g.,
        > input shape : 2
        > output shape: [1, 2, 1]

        > input shape : [2]
        > output shape: [1, 2, 1]

        > input shape : [1, 2, 1]
        > output shape: [1, 2, 1]
    """
    if isinstance(shape, int):
        shape = [1, shape, 1]
    elif isinstance(shape, (list, tuple)):
        if len(shape) == 1:
            shape = [1, shape[0], 1]
        elif len(shape) != 3:
            raise ValueError('require input shape {}[batch-size,'
                             ' cols, channels]{}, given {}'
                             .format(colors.fg.green, colors.reset,
                                     colors.red(shape)))
    else:
        raise TypeError('shape require {}int/list/tuple{} type, given `{}`'
                        .format(colors.fg.green, colors.reset,
                                colors.red(type(shape))))
    return shape


def norm_input_2d(shape):
    """ norm input for 2d convolving operation
        generally called by norm kernel-shape and stride
        e.g.,
        > input shape : 2
        > output shape: [1, 2, 2, 1]

        > input shape : [2]
        > output shape: [1, 2, 2, 1]

        > input shape : [2, 2]
        > output shape: [1, 2, 2, 1]

        > input shape : [1, 2, 2, 1]
        > output shape: [1, 2, 2, 1]
    """
    if isinstance(shape, int):
        shape = [1, shape, shape, 1]
    elif isinstance(shape, (list, tuple)):
        if len(shape) == 1:
            shape = [1, shape[0], shape[0], 1]
        elif len(shape) == 2:
            shape = [1, shape[0], shape[1], 1]
        elif len(shape) != 4:
            raise ValueError('require input shape {}[batch-size, '
                             'rows, cols, channels]{}, given {}'
                             .format(colors.fg.green, colors.reset,
                                     colors.red(shape)))
    else:
        raise TypeError('shape requires {}int/list/tuple{} type, given `{}`'
                        .format(colors.fg.green, colors.reset,
                                colors.red(type(shape))))
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
                             'depths, rows, cols, channels]{}, given {}'
                             .format(colors.fg.green, colors.reset,
                                     colors.red(input_shape)))
    else:
        raise TypeError('shape requires {}int/list/tuple{} type, given `{}`'
                        .format(colors.fg.green, colors.reset,
                                colors.red(type(kshape))))
