import inspect
from contextlib import contextmanager
from ..ops import helper


_CONTEXT_DEFAULTS = []


@contextmanager
def defaults(**kwargs):
    global _CONTEXT_DEFAULTS
    _CONTEXT_DEFAULTS = [kwargs] + _CONTEXT_DEFAULTS
    yield kwargs
    del _CONTEXT_DEFAULTS[-1]


""" layers decorator
        layers decorated using @layers must have the spec:
        fun layername(inputs, [...], reuse, name) => x[, output_shape]
"""
def layers(fun):
    def _get(names, args, kwargs, parameters, defaults=None):
        _names = names
        if not isinstance(_names, (list, tuple)):
            _names = [_names]
        if not isinstance(defaults, (list, tuple)):
            defaults = [defaults] * len(_names)
        if len(defaults) != len(_names):
            raise ValueError('`names` and `defaults` have different length.'
                             ' given {} vs {}'
                             .format(len(_names), len(defaults)))
        length = len(args)
        ans = []
        for name, default in zip(_names, defaults):
            var = kwargs.get(name, default)
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
        helper.print_layer(inputs, outputs, fun.__name__, reuse, name)
        return x
    return _wrap
