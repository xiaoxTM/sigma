import inspect
from contextlib import contextmanager

_CONTEXT_DEFAULTS = []

@contextmanager
def defaults(**kwargs):
    global _CONTEXT_DEFAULTS
    _CONTEXT_DEFAULTS = [kwargs] + _CONTEXT_DEFAULTS
    yield kwargs
    del _CONTEXT_DEFAULTS[-1]

def layers(fun):
    def _wrap(*args, **kwargs):
        parameters = inspect.getfullargspec(fun)[0]
        signature = inspect.signature(fun)
        for parameter in parameters:
            if parameter not in kwargs.keys():
                for ctx in _CONTEXT_DEFAULTS:
                    if parameter in ctx.keys():
                        kwargs[parameter] = ctx[parameter]
                        break
        if signature.return_annotation is signature.empty:
            return fun(*args, **kwargs)
        else:
            fun(*args, **kwargs)
    return _wrap


graph = None
