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

def normalize_name(x):
    if x is None:
        raise ValueError('`x` must be non None for name normalize')
    if not isinstance(x, str):
        raise TypeError('`x` muset be string for name normalize. given `{}`'.format(type(x)))
    if x[0] == '.':
        x = '.'+x
    if x[-1] == '.':
        x = x+'.'
    return x

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
            return normalize_name('{}-{}'.format(*list(name_maps.items())[index]))
        else:
            if index is None:
                if x not in name_maps.keys():
                    name_maps[x] = -1
                nid = name_maps[x] + 1
                name_maps[x] = nid
                return normalize_name('{}-{}'.format(x, nid))
            else:
                if x not in name_maps.keys():
                    raise ValueError('`{}` not in name maps for indexing {}'
                                     .format(x, index))
                size = name_maps[x] + 1
                nid = (size + index) % size
                return normalize_name('{}-{}'.format(x, nid))
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
    name = normalize_name(name)
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
        ops_scope, _, _ = assign_scope(name, scope, ltype, reuse)
        yield ops_scope
    else:
        yield


def is_tensor(x):
    return core.is_tensor(x)


def depth(x):
    return core.shape(x)[core.caxis]


def feature_dims(x):
    dims = list(range(core.rank(x)))
    shape.pop(core.caxis)
    shape.pop(0)
    return shape

def split_scope(name):
    def _split_scope(n):
        scope = n.rsplit(':', 1)
        if len(scope) == 2:
            if scope[1].isnumeric():
                # in case of name:index
                scope = scope[0].rsplit(':', 1)
            if len(scope) == 2:
                scope = scope[0]
            else:
                scope = 'None'
        else:
            scope = 'None'
        return scope
    if isinstance(name, str):
        return _split_scope(name)
    elif isinstance(name, (tuple, list)):
        return list(map(_split_scope, name))

''' names is list of string who has form of
    [scope]+, [layer-name, layer-type,]? variable-name
'''
def concat_scope_and_name(names):
    def _concat_scope_and_name(name):
        if not isinstance(name, list):
            raise TypeError('name type must be list. given {}'
                            .format(type(name)))
        length = len(name)
        if length < 1 or length > 2:
            raise ValueError('name must have length in [1, 2], given {}'
                             .format(length))
        elif length == 1:
            # name = [layer-name]
            return '{}'.format(name[0])
        else: #length == 2
            # name = [scope+, layer-name]
            return '{}-{}'.format(name[0], name[1])
    if isinstance(names, list):
        return _concat_scope_and_name(names)
    elif isinstance(names, tuple):
        tuples = tuple(map(_concat_scope_and_name, names))
        if len(tuples) == 1:
            return tuples[0]
        return tuples

def split_name(names, nameonly=True):
    """ split variable name (or say, remove variable index)
        generally, `names` are a list of :
            [scope/]*/.layer-name./layer-type/[subspace/*/]variable-name:index
        return [scope(s),] layer-name
    """
    def _split(name):
        if name.find('/.') == -1:
            # no scope
            # .layer-name./layer-type/[subspace*/]variable-name:index
            # ==> layer-name
            name = name.split('./')[0]
            if name[0] == '.':
                name = name[1:]
            name = [name]
        else:
            # [scope/]*/.layer-name./layer-type/[subspace*/]variable-name:index
            # ==> scope/*, layer-name
            name = name.split('./')[0].split('/.')
        if nameonly is True:
            return name[-1]
        else:
            return name
    if isinstance(names, str):
        return _split(names)
    elif isinstance(names, (tuple, list, np.ndarray)):
        return tuple(map(_split, names))
    else:
        raise TypeError('name must be tuple/list/np.ndarray. given {}'
                        .format(type(names)))


def scope_name(names):
    name_lists = split_name(names, False)
    return concat_scope_and_name(name_lists)


""" normalize axis given tensor shape
    for example: tensor shape [batch size, rows, cols, channels], axis = -1
        return axis=3
"""
def normalize_axes(shape, axis=core.caxis):
    if not isinstance(shape, (list, tuple)):
        raise TypeError('shape must be list/tuple, given {}[{}]'
                        .format(colors.red(type(shape)), shape))
    input_len = len(shape)
    if isinstance(axis, int):
        axis = (axis + input_len) % input_len
    elif isinstance(axis, (list, tuple)):
        axis = map(lambda x:(x + input_len) % input_len, axis)
    else:
        raise TypeError('axis must be int or list/tuple, given {}[{}]'
                        .format(colors.red(type(axis)), axis))
    return axis


def get_output_shape(input_shape, channels, kshape, stride, padding):
    """ get the corresponding output shape given tensor shape
        Attributes
        ==========
            input_shape : list / tuple
                          input tensor shape
            channels : int
                    number of output feature maps
            kshape : list / tuple
                     kernel shape for convolving operation
            stride : list / tuple
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
    out_shape = list(input_shape[:])
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
    out_shape[-1] = channels
    return out_shape


def norm_input_shape(input_tensor):
    """ get the input shape
        if the first axis is None,
        will be assigned to Tensor as batch_size
    """
    input_shape = core.shape(input_tensor)
    if len(input_shape) != 0 and input_shape[0] is None:
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

def check_shape_consistency(input_shapes):
    if not isinstance(input_shapes, (list, tuple)):
        raise TypeError('requires inputs as '
                        '{}list / tpule{}, given {}'
                        .format(colors.fg.green, colors.reset,
                                colors.red(type(input_shapes))))
    elif len(input_shapes) < 2:
        raise ValueError('requires at least {}two{} inputs, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(len(input_shapes))))
    output_shape = input_shapes[0]
    check_input_shape(output_shape)
    for idx, ip in enumerate(input_shapes[1:]):
        check_input_shape(ip)
        # ignore the batch-size dimension in case of value `None`
        if len(output_shape) == 0:
            if len(ip) != 0:
                raise ValueError('shape of {}-input '
                                 'has length {} while '
                                 '0-input is scalar'
                                 .format(colors.red(idx+1),
                                         colors.red(len(ip)))
                                )
        if not np.all(output_shape[1:] == ip[1:]):
            raise ValueError('shape of {}-input differ '
                             'from first one. {} vs {}'
                             .format(colors.red(idx+1),
                                     colors.red(output_shape),
                                     colors.green(ip)))

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
    return list(shape)


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
    return list(shape)


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
                                     colors.red(shape)))
    else:
        raise TypeError('shape requires {}int/list/tuple{} type, given `{}`'
                        .format(colors.fg.green, colors.reset,
                                colors.red(type(shape))))
    return list(shape)

def split_inputs(inputs, types=(list, tuple), allow_none=-1):
    if not isinstance(inputs, types):
        raise TypeError('input must be {}. given {}'.format(colors.green(types), colors.red(type(inputs))))
    for i, v in enumerate(inputs):
        if i != allow_none and v is None:
            raise ValueError('{}-th element is None, which is not allowed'.format(colors.red(i)))
    return inputs
