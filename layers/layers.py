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
    def _wrap(inputs, *args, **kwargs):
        parameters = inspect.getfullargspec(fun)[0]
        signature = inspect.signature(fun)
        for parameter in parameters:
            if parameter not in kwargs.keys():
                for ctx in _CONTEXT_DEFAULTS:
                    if parameter in ctx.keys():
                        kwargs[parameter] = ctx[parameter]
                        break
        reuse = kwargs.get('reuse', False)
        kwargs['name'] = helper.name_assign(kwargs.get('name', None),
                                            fun.__name__,
                                            reuse)
        x = fun(inputs, *args, **kwargs)
        outputs = x
        if isinstance(x, (list, tuple)):
            outputs = x[0]
        helper.print_layer(inputs, outputs, fun.__name__, reuse, kwargs['name'])
        return x
    return _wrap


graph = False
