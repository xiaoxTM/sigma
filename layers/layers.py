import inspect
import functools
from contextlib import contextmanager
from ..ops import helper, core
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


_CONTEXT_DEFAULTS_ = {}


__layers__ = {'actives': ['crelu',
                          'relu',
                          'relu6',
                          'elu',
                          'selu',
                          'leaky_relu',
                          'softmax',
                          'softplus',
                          'softsign',
                          'sigmoid',
                          'tanh',
                          'linear'],
              'base': ['flatten', 'reshape'],
              'convs': ['fully_conv',
                        'dense',
                        'conv1d',
                        'conv2d',
                        'conv3d',
                        'soft_conv2d',
                        'deconv2d',
                        'sepconv2d'],
              'merge': ['concat', 'add', 'mul'],
              'norms': ['instance_norm', 'batch_norm', 'dropout'],
              'pools': ['avg_pool2d',
                        'avg_pool2d_global',
                        'max_pool2d',
                        'max_pool2d_global']
             }


_colormaps = {'actives' : 'Magenta',
              'base' : 'Red',
              'convs' : 'Blue',
              'merge' : 'Cyan',
              'norms' : 'Cyan',
              'pools' : 'Gold',
              'other' : 'Yellow'
             }


def _layer2color(lname):
    for k,v in __layers__.items():
        if lname in v:
            return _colormaps[k]
    return _colormaps['other']


@contextmanager
def defaults(*args, **kwargs):
    """ set defaults to context
        Attributes
        ----------
            args : callable
                   function list
            kwargs : dict
                     key-value pairs
    """
    for idx, arg in enumerate(args):
        if not callable(arg):
            raise TypeError('args at {}-th is not callable. given {}'
                            .format(idx, arg))
    list_args = list(map(lambda x:inspect.signature(x.__wrapped__), args))
    global _CONTEXT_DEFAULTS_
    # print([arg.__wrapped__ for arg in args])
    context_keys = _CONTEXT_DEFAULTS_.keys()
    for k,v in kwargs.items():
        value = [v, list_args]
        if k in context_keys:
            # append value (and functions) to exists list
            _CONTEXT_DEFAULTS_[k].append(v)
        else:
            _CONTEXT_DEFAULTS_[k] = [value]
    # print('context:', _CONTEXT_DEFAULTS_)
    yield _CONTEXT_DEFAULTS_
    for k in kwargs.keys():
        # remove the latest value
        _CONTEXT_DEFAULTS_[k].pop(-1)
        # if left empty list for key `k`, clear dict
        if len(_CONTEXT_DEFAULTS_[k]) == 0:
            _CONTEXT_DEFAULTS_.pop(k)


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
def _print_layer(inputs, outputs, typename, reuse, name, **kwargs):
    if status.graph is not None:
        if not reuse:
            if name is None:
                raise ValueError('name is not given')
            if isinstance(inputs, (list, tuple)):
                input_shape = [shape(x) for x in inputs]
                inputname = [helper.name_normalize(x.name) for x in inputs]
            else:
                input_shape = [core.shape(inputs)]
                inputname = [helper.name_normalize(inputs.name)]
            output_shape = core.shape(outputs)
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
    @functools.wraps(fun)
    def _wrap(*args, **kwargs):
        # parameters = inspect.getfullargspec(fun)[0]
        signature = inspect.signature(fun)
        items = list(signature.parameters.items())
        # merge args into kwargs
        for idx, arg in enumerate(args):
            kwargs[items[idx][0]] = arg
        for name, parameter in items:
            if name not in kwargs.keys():
                # if parameter and the corresponding function in context
                if name in _CONTEXT_DEFAULTS_.keys():
                    lists = _CONTEXT_DEFAULTS_[name]
                    found = False
                    # find fron context traversely
                    for (v,funcs) in lists:
                        if len(funcs) == 0 or fun in funcs:
                            kwargs[name] = v
                            found = True
                            break
                    if not found:
                        if parameter.default is parameter.empty:
                            raise LookupError('`{}` requires value.'.format(name))
                        kwargs[name] = parameter.default
                else:
                    if parameter.default is parameter.empty:
                        raise LookupError('`{}` requires value.'.format(name))
                    kwargs[name] = parameter.default
        name, reuse = kwargs.get('name', None), kwargs.get('reuse', False)
        if name is None:
            if reuse:
                name = helper.dispatch_name(fun.__name__, -1)
            else:
                name = helper.dispatch_name(fun.__name__)
            kwargs['name'] = name
        x = fun(**kwargs)
        outputs = x
        if isinstance(x, (list, tuple)):
            outputs = x[0]
        inputs = kwargs.pop('inputs')
        kwargs.pop('reuse')
        kwargs.pop('name')
        # print('func:', fun.__name__)
        # print('kwargs keys:', kwargs.keys())
        _print_layer(inputs, outputs, fun.__name__, reuse, name, **kwargs)
        return x
    return _wrap
