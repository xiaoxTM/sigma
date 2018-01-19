import tensorflow as tf
from .. import colors, status
import numpy as np
import copy
import collections
import os.path

try:
    import pydot_ng as pydot
except ImportError:
    try:
        import pydotplus as pydot
    except ImportError:
        try:
            import pydot
        except ImportError:
            pydot = None


layer_actives = ['crelu', 'relu', 'relu6', 'elu', 'selu', 'leaky_relu',
                 'softmax', 'softplus', 'softsign', 'sigmoid', 'tanh', 'linear']
layer_base = ['flatten', 'reshape']
layer_convolutional = ['fully_conv', 'dense', 'conv1d', 'conv2d', 'conv3d',
                       'soft_conv2d', 'deconv2d']
layer_merge = ['concat', 'add', 'mul']
layer_normalization = ['instance_norm', 'batch_norm', 'dropout']
layer_pools = ['avg_pool2d', 'avg_pool2d_global',
               'max_pool2d', 'max_pool2d_global']


def layer2color(layername):
    if layername in layer_actives:
        return 'Magenta'
    elif layername in layer_base:
        return 'Red'
    elif layername in layer_convolutional:
        return 'Blue'
    elif layername in layer_merge:
        return 'Cyan'
    elif layername in layer_normalization:
        return 'Green'
    elif layername in layer_pools:
        return 'Gold'
    else:
        return 'Yellow'


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
                    raise ValueError('{} not in name maps for indexing {}'
                                     .format(x, index))
                size = name_maps[x] + 1
                nid = (size + index) % size
                return '{}-{}'.format(x, nid)
    return _name_space


dispatch_name = name_space()


# def name_assign(name, default, reuse):
#     if name is None:
#         name = dispatch_name(default, reuse)
#     return name


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
    layer = '{}/{}'.format(name, ltype)
    if scope is None:
        ops_scope = tf.name_scope('{}'.format(layer))
    else:
        ops_scope = tf.name_scope('{}/{}'
                                  .format(scope, layer))
    return ops_scope, layer


def is_tensor(x):
    return tf.contrib.framework.is_tensor(x)


def shape(x):
    return x.get_shape().as_list()


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


def export_graph(filename, ext):
    if pydot is None:
        raise ImportError('Import pydot failed. make sure pydot is installed')
    if not isinstance(status.graph, pydot.Dot):
        raise TypeError('graph model is not built. graph: {}'.format(status.graph))
    if ext is None:
        _, ext = os.path.splitext(filename)
    if ext is None:
        ext = 'png'
    else:
        ext = ext[1:]
    status.graph.write(filename, format=ext)


""" if not reuse:
        print('{}{}{} \t\t=>`[{} | {}]`=> \t\t{}{}{}'
              .format(colors.fg.green, input_shape, colors.reset,
                      name if name is not None else 'concat', 'concat',
                      colors.fg.red, output, colors.reset))
"""
def print_layer(inputs, outputs, typename, reuse, name):
    if status.graph is not None:
        if not reuse:
            if name is None:
                raise ValueError('name is not given')
            if isinstance(inputs, (list, tuple)):
                input_shape = [shape(x) for x in inputs]
                inputname = [name_normalize(x.name) for x in inputs]
            else:
                input_shape = [inputs.get_shape().as_list()]
                inputname = [name_normalize(inputs.name)]
            output_shape = outputs.get_shape().as_list()
            outputname = name_normalize(outputs.name)
            if status.graph is False:
                if len(inputname) == 1:
                    inputname = inputname[0]
                    input_shape = input_shape[0]
                print('{}{}{}{} \t\t=>`{}[{} | {}]{}`=> \t\t{}{}{}{}'
                      .format(inputname, colors.fg.green, input_shape, colors.reset,
                              colors.fg.blue, name, typename, colors.reset,
                              outputname, colors.fg.red, output_shape, colors.reset))
            elif status.graph is True or isinstance(status.graph, pydot.Dot):
                if pydot is None:
                    raise ImportError('Import pydot failed. make sure pydot is installed')
                if status.graph is True:
                    dot = pydot.Dot()
                    dot.set('rankdir', 'TB')
                    dot.set('concentrate', True)
                    dot.set_node_defaults(shape='record')
                    for iname, ishape in zip(inputname, input_shape):
                        label = '%s\n|{{%s}}' % (iname, ishape)
                        node = pydot.Node(iname,
                                          label=label)
                        dot.add_node(node)
                    status.graph = dot
                ishape = input_shape
                if len(ishape) == 1:
                    ishape = ishape[0]
                label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (name,
                                                               ishape,
                                                               output_shape)
                color = layer2color(typename)
                status.graph.add_node(pydot.Node(outputname,
                                                 label=label,
                                                 fillcolor=color,
                                                 style='filled'))
                for iname in inputname:
                    status.graph.add_edge(pydot.Edge(iname, outputname))


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
