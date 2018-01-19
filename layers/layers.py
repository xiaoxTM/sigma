import inspect
from contextlib import contextmanager
from ..ops import helper
from .. import status

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


_CONTEXT_DEFAULTS = []


layer_actives = ['crelu', 'relu', 'relu6', 'elu', 'selu', 'leaky_relu',
                 'softmax', 'softplus', 'softsign', 'sigmoid', 'tanh', 'linear']
layer_base = ['flatten', 'reshape']
layer_convolutional = ['fully_conv', 'dense', 'conv1d', 'conv2d', 'conv3d',
                       'soft_conv2d', 'deconv2d', 'sepconv2d']
layer_merge = ['concat', 'add', 'mul']
layer_normalization = ['instance_norm', 'batch_norm', 'dropout']
layer_pool = ['avg_pool2d', 'avg_pool2d_global',
               'max_pool2d', 'max_pool2d_global']


_colormaps = {'actives' : 'Magenta',
              'base' : 'Red',
              'convolutional' : 'Blue',
              'merge' : 'Cyan',
              'normalization' : 'Cyan',
              'pool' : 'Gold',
              'other' : 'Yellow'
             }


def _layer2color(layername):
    if layername in layer_actives:
        return _colormaps['actives']
    elif layername in layer_base:
        return _colormaps['base']
    elif layername in layer_convolutional:
        return _colormaps['convolutional']
    elif layername in layer_merge:
        return _colormaps['merge']
    elif layername in layer_normalization:
        return _colormaps['normalization']
    elif layername in layer_pool:
        return _colormaps['pool']
    else:
        return _colormaps['other']


@contextmanager
def defaults(**kwargs):
    global _CONTEXT_DEFAULTS
    _CONTEXT_DEFAULTS = [kwargs] + _CONTEXT_DEFAULTS
    yield kwargs
    del _CONTEXT_DEFAULTS[-1]


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
def _print_layer(inputs, outputs, typename, reuse, name):
    if status.graph is not None:
        if not reuse:
            if name is None:
                raise ValueError('name is not given')
            if isinstance(inputs, (list, tuple)):
                input_shape = [shape(x) for x in inputs]
                inputname = [helper.name_normalize(x.name) for x in inputs]
            else:
                input_shape = [inputs.get_shape().as_list()]
                inputname = [helper.name_normalize(inputs.name)]
            output_shape = outputs.get_shape().as_list()
            outputname = helper.name_normalize(outputs.name)
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
                color = _layer2color(typename)
                status.graph.add_node(pydot.Node(outputname,
                                                 label=label,
                                                 fillcolor=color,
                                                 style='filled'))
                for iname in inputname:
                    status.graph.add_edge(pydot.Edge(iname, outputname))


""" layers decorator
        layers decorated using @layers must have the spec:
        fun layername(inputs, [...], reuse, name) => x[, output_shape]
"""
def layers(fun):
    def _get(names, args, kwargs, parameters, values=None):
        _names = names
        if not isinstance(_names, (list, tuple)):
            _names = [_names]
        if not isinstance(values, (list, tuple)):
            values = [values] * len(_names)
        if len(values) != len(_names):
            raise ValueError('`names` and `defaults` have different length.'
                             ' given {} vs {}'
                             .format(len(_names), len(values)))
        length = len(args)
        ans = []
        for name, value in zip(_names, values):
            var = kwargs.get(name, value)
            var_idx = parameters.index(name)
            if args is not None and length >= var_idx:
                var = args[var_idx-1]
            ans.append(var)
        if not isinstance(names, (tuple, list)):
            return ans[0]
        return ans

    def _set(names, args, kwargs, parameters, values=None):
        if not isinstance(names, (list, tuple)):
            names = [names]
        if not isinstance(values, (list, tuple)):
            values = [values] * len(names)
        if len(values) != len(names):
            raise ValueError('`names` and `values` have different length.'
                             ' given {} vs {}'
                             .format(len(names), len(values)))
        length = len(args)
        for name, value in zip(names, values):
            idx = parameters.index(name)
            if args is not None and length >= idx:
                args[idx-1] = value
            else:
                kwargs[name] = value
        return args, kwargs

    def _wrap(inputs, *args, **kwargs):
        parameters = inspect.getfullargspec(fun)[0]
        signature = inspect.signature(fun)
        for parameter in parameters:
            if parameter not in kwargs.keys():
                for ctx in _CONTEXT_DEFAULTS:
                    if parameter in ctx.keys():
                        kwargs[parameter] = ctx[parameter]
                        break
        name, reuse = _get(['name', 'reuse'], args, kwargs, parameters, [None, False])
        if name is None:
            if reuse:
                name = helper.dispatch_name(fun.__name__, -1)
            else:
                name = helper.dispatch_name(fun.__name__)
            args, kwargs = _set('name', args, kwargs, parameters, name)
        # print('name:', name)
        x = fun(inputs, *args, **kwargs)
        outputs = x
        if isinstance(x, (list, tuple)):
            outputs = x[0]
        _print_layer(inputs, outputs, fun.__name__, reuse, name)
        return x
    return _wrap
