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

import inspect
import functools
from contextlib import contextmanager
from .. import ops, colors
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


_CONTEXT_DEFAULTS_ = {}

# None  : print no network information
# False : print network information to terminal [default]
# True  : print network information to graph
__graph__ = False
# if __graph__ is not None,
#   False : print no parameters of layers
#   True  : print all parameters of layers
__details__ = False


__defaults__ = {'padding' : 'valid',
                'summary' : 'histogram',
                'weight_initializer' : 'glorot_uniform',
                'weight_regularizer' : None,
                'bias_initializer' : 'zeros',
                'bias_regularizer' : None
               }

# __layers__ = ['squash', 'crelu', 'relu', 'relu6', 'elu', 'selu', 'leaky_relu',
#               'softmax', 'softplus', 'softsign', 'sigmoid', 'tanh', 'linear',
#               'embedding', 'flatten', 'reshape', 'expand_dims', 'maskout',
#               'input_spec', 'label_spec', 'random_spec',
#              ]


__modules__ = {'actives': ['crelu',
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
              'base': ['flatten',
                       'reshape',
                       'expand_dims',
                       'norm',
                       'maskout'],
              'convs': ['fully_conv',
                        'fully_connected',
                        'dense',
                        'dot',
                        'conv1d',
                        'conv2d',
                        'conv3d',
                        'soft_conv2d',
                        'deconv2d',
                        'sepconv2d'],
              'merge': ['concat',
                        'add',
                        'mul'],
              'norms': ['instance_norm',
                        'batch_norm',
                        'dropout'],
              'pools': ['avg_pool2d',
                        'avg_pool2d_global',
                        'max_pool2d',
                        'max_pool2d_global']
             }


__colormaps__ = {'actives' : 'Magenta',
                 'base' : 'Red',
                 'convs' : 'Blue',
                 'merge' : 'Cyan',
                 'norms' : 'Cyan',
                 'pools' : 'Gold',
                 'other' : 'Yellow'
                 }

def split_inputs(inputs):
    """ split inputs into two parts according to the type of inputs:
        - inputs
        - output
    """
    if isinstance(inputs, (list, tuple)):
        if len(inputs) == 1:
            inputs, labels = inputs[0], None
        elif len(inputs) == 2:
            inputs, labels = inputs
        else:
            raise ValueError('`inputs` as list must have length of 1 or 2.'
                             ' given {}'.format(len(inputs)))
    elif isinstance(inputs, dict):
        inputs = inputs['logits']
        labels = inputs.get('labels', None)
    else:
        raise TypeError('`inputs` must be list / tuple / dict.'
                        ' given {}'.format(type(inputs)))

    return [inputs, labels]


# interstatus for visualization
# must be one of [None, True, False]
#    None: no output
#    True: print layers to graph
#          will change to instance of pydot.Dot
#    False:print layers to terminal
# for channels-last format
#   kshape shape: [row, col, ins, outs] for 2d
# for channel-first (without batch dimension) format
#   kshape shape: [ins, outs, row, col] for 2d
def set_print(mode: bool=True, details: bool=None):
    global __graph__
    global __details__
    if mode is None or isinstance(mode, bool):
        __graph__ = mode
    else:
        raise TypeError('mode must be None or True/False, given {}'
                        .format(mode))
    if details is not None:
        if isinstance(details, bool):
            __details__ = details
        else:
            raise TypeError('`details` must be bool type, given {}'
                            .format(type(details)))


def set_defaults(key_values: dict):
    if key_values is not None:
        if not isinstance(key_values, dict):
            raise TypeError('`key_values` for set_defaults must be dict type.'
                            'given {}'.format(type(key_values)))
        global __defaults__
        keys = __defaults__.keys()
        for key, value in key_values.items():
            if key not in keys:
                raise KeyError('key `{}` not in __defaults__'.format(key))
            __defaults__[key] = value


def _layer2color(lname: dict) -> str:
    for k,v in __modules__.items():
        if lname in v:
            return __colormaps__[k]
    return __colormaps__['other']


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

    funcs = list(map(lambda x:inspect.signature(x.__wrapped__), args))
    global _CONTEXT_DEFAULTS_
    context_keys = _CONTEXT_DEFAULTS_.keys()
    for k,v in kwargs.items():
        value = [v, funcs]
        if k in context_keys:
            # append value (and functions) to exists list
            _CONTEXT_DEFAULTS_[k].append(v)
        else:
            _CONTEXT_DEFAULTS_[k] = [value]
    yield _CONTEXT_DEFAULTS_
    for k in kwargs.keys():
        # remove the latest value
        _CONTEXT_DEFAULTS_[k].pop(-1)
        # if left empty list for key `k`, clear dict
        if len(_CONTEXT_DEFAULTS_[k]) == 0:
            _CONTEXT_DEFAULTS_.pop(k)


def print_value(x):
    if x is None:
        return 'None'
    elif inspect.isclass(x):
        return x.name
    elif callable(x):
        return x.__name__
    elif ops.core.is_tensor(x):
        return x.name
    elif isinstance(x, (list, tuple)):
        return str([print_value(v) for v in x])
    return str(x)


def export_graph(filename, ext=None):
    global __graph__
    if pydot is None:
        raise ImportError('Importing pydot failed. make sure pydot is installed')
    if not isinstance(__graph__, pydot.Dot):
        raise TypeError('graph model is not built. graph: {}'.format(__graph__))
    if ext is None:
        _, ext = os.path.splitext(filename)
    if ext is None:
        ext = 'png'
    else:
        ext = ext[1:]
    __graph__.write(filename, format=ext)


def graph_has_node(graph, name):
    if isinstance(graph, bool):
        return False
    if isinstance(name, (tuple, list)):
        return False
    name = name.replace(':', '-')
    if len(graph.get_node(name)) == 0:
        return False
    return True

def graph_has_edge(graph, input_name, output_name):
    if isinstance(graph, bool):
        return False
    if isinstance(input_name, (tuple, list)):
        return False
    input_name = input_name.replace(':', '-')
    output_name = output_name.replace(':', '-')
    if len(graph.get_edge(input_name, output_name)) == 0:
        return False
    return True

""" if not reuse:
        print('{}{}{} \t=>`[{} | {}]`=> \t{}{}{}'
              .format(colors.fg.green, input_shape, colors.reset,
                      name if name is not None else 'concat', 'concat',
                      colors.fg.red, output, colors.reset))
"""
def _print_layer(inputs, outputs, typename, reuse, name, scope, **kwargs):
    """ print each layer
        parameters:
            @name: layer name
            @typename: layer type name
    """
    # //FUTURE: print details of each layer. e.g., parameters of each layer. For graph ONLY
    global __graph__
    if __graph__ is not None:
        is_input_layer = False
        if name is None:
            raise ValueError('name is not given')
        if isinstance(inputs, (list, tuple)):
            if ops.helper.is_tensor(inputs[0]):
                input_shape = [ops.core.shape(x) for x in inputs]
                # input_name is a tuple of strings
                input_name = list(ops.helper.scope_name([x.name for x in inputs]))
            else:
                is_input_layer = True
                # inputs is a list of int to represent the shape of input tensor
                input_shape = inputs
                # input_name is a string
                input_name = ops.helper.scope_name(name)
        elif ops.helper.is_tensor(inputs):
            input_shape = ops.core.shape(inputs)
            # input_name is a string
            input_name = ops.helper.scope_name(inputs.name)
        else:
            raise TypeError('inputs must be list/tuple or Tensor. given {}'
                            .format(type(inputs)))
        # outputname is a string
        output_name = ops.helper.scope_name(outputs.name)
        has_edge = graph_has_edge(__graph__, input_name, output_name)
        if not reuse or not has_edge:
            output_shape = ops.core.shape(outputs)
            if __graph__ is False:
                if not is_input_layer:
                    # if isinstance(inputname, tuple) and len(inputname) == 1:
                    #     inputname = inputname[0]
                    #     input_shape = input_shape[0]
                    # input_name_str = ops.helper.concat_scope_and_name(inputname)
                    input_shape_str = str(input_shape)
                    input_name_str = str(input_name).replace("'","")
                    print('{}{} \t=>`{}[{} | {}]{}`=> \t{}{}'
                          .format(input_name_str,
                                  colors.green(input_shape_str),
                                  colors.fg.blue, name, typename, colors.reset,
                                  output_name, colors.red(output_shape)))
                    if __details__ is True:
                        length = len(input_name_str) + len(input_shape_str)
                        for key, value in kwargs.items():
                            print('{} \t\t  {}:{}'
                                  .format(' '*length,
                                          key, colors.blue(print_value(value))))
            elif __graph__ is True or isinstance(__graph__, pydot.Dot):
                if pydot is None:
                    raise ImportError('Import pydot failed. \
                                      make sure pydot is installed')
                if __graph__ is True:
                    dot = pydot.Dot()
                    dot.set('rankdir', 'TB')
                    dot.set('concentrate', True)
                    dot.set_node_defaults(shape='record')
                    __graph__ = dot
                input_scope = str(ops.helper.split_scope(input_name)).replace("'","")
                output_scope = str(ops.helper.split_scope(output_name)).replace("'","")
                if is_input_layer:
                    if isinstance(input_name, (list, tuple)):
                        raise TypeError('input layer cannot be list/tuple of shapes')
                    output_name = output_name.replace(':', '-')
                    label = '%s\n|{{%s}}|{{%s}}' % (output_name, output_shape, input_scope)
                    node = pydot.Node(output_name, label=label)
                    __graph__.add_node(node)
                else:
                    if __details__ is True:
                        # input_scope = ops.helper.split_scope(input_name)
                        # output_scope = ops.helper.split_scope(output_name)
                        # if isinstance(input_scope, str):
                        #     input_scope = [input_scope]
                        parameters = None
                        for key, value in kwargs.items():
                            value = print_value(value)
                            # get rid of '<' and '>' otherwise can not print parameters
                            value = value.replace('<', '[').replace('>', ']').replace("'","")
                            if parameters is None:
                                parameters = '{}:{}'.format(key, value)
                            else:
                                parameters = '{}\n{}:{}'.format(parameters,
                                                                key, value)
                        label = '{{%s\n|{input:|output:}|{{%s}|{%s}}|{{%s}|{%s}}}|{{%s}}}' % (name,
                                                                                  str(input_shape).replace("'",""),
                                                                                  output_shape,
                                                                                  input_scope,
                                                                                  output_scope,
                                                                                  parameters)
                    else:
                        label = '%s\n|{input:|output:}|{{%s}|{%s}}|{{%s}|{%s}}' % (name,
                                                                       str(input_shape).replace("'",""),
                                                                       output_shape,
                                                                       input_scope,
                                                                       output_scope)
                    color = _layer2color(typename)
                    output_name = output_name.replace(':', '-')
                    if not graph_has_node(__graph__, output_name):
                        __graph__.add_node(pydot.Node(output_name,
                                                      label=label,
                                                      fillcolor=color,
                                                      style='filled'))
                    if isinstance(input_name, str):
                        input_name = [input_name]
                        input_shape = [input_shape]
                    for iname, ishape in zip(input_name, input_shape):
                        iname = iname.replace(':', '-')
                        if reuse and not has_edge:
                            __graph__.add_edge(pydot.Edge(iname, output_name, style='dashed'))
                        else:
                            __graph__.add_edge(pydot.Edge(iname, output_name))


def assign(fun, *args, **kwargs):
    signature = inspect.signature(fun)
    items = list(signature.parameters.items())
    # merge args into kwargs
    for idx, arg in enumerate(args):
        kwargs[items[idx][0]] = arg
    ctx_keys = _CONTEXT_DEFAULTS_.keys()
    for name, parameter in items:
        if parameter.kind != parameter.VAR_KEYWORD \
           and parameter.kind != parameter.VAR_POSITIONAL:
            if name not in kwargs.keys():
                # if parameter and the corresponding function in context
                # print('context keys:', ctx_keys)
                if name in ctx_keys:
                    lists = _CONTEXT_DEFAULTS_[name]
                    # print('lists:', lists)
                    found = False
                    # find from context traversely
                    for (v,funcs) in lists:
                        if len(funcs) == 0 or signature in funcs:
                            kwargs[name] = v
                            found = True
                            break
                    if not found:
                        if parameter.default is parameter.empty:
                            raise LookupError('`{}` requires value.'
                                              .format(name))
                        kwargs[name] = parameter.default
                else:
                    if parameter.default is parameter.empty:
                        raise LookupError('`{}` requires value.'.format(name))
                    kwargs[name] = parameter.default
    return kwargs


""" decorator to make function available
    to core.defaults() as function lists
"""
def defaultable(fun):
    @functools.wraps(fun)
    def _wrap(*args, **kwargs):
        kwargs = assign(fun, *args, **kwargs)
        return fun(**kwargs)
    return _wrap


""" layer decorator
        layer decorated using @layer must have the spec:
        fun layername(inputs, [...], reuse, name) => x[, output_shape]
"""
def layer(fun):
    @functools.wraps(fun)
    def _wrap(*args, **kwargs):
        kwargs = assign(fun, *args, **kwargs)
        name, reuse = kwargs.get('name', None), kwargs.get('reuse', False)
        if name is None:
            if reuse:
                name = ops.helper.dispatch_name(fun.__name__, -1)
            else:
                name = ops.helper.dispatch_name(fun.__name__)
            kwargs['name'] = name
        x = fun(**kwargs)
        outputs = x
        if isinstance(x, (list, tuple)):
            outputs = x[0]
        inputs = kwargs.pop('inputs')
        kwargs.pop('reuse')
        kwargs.pop('name')
        scope = kwargs.pop('scope')
        _print_layer(inputs, outputs, fun.__name__, reuse,
                     name, scope, **kwargs)
        return x
    return _wrap
