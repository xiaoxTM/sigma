import inspect
from contextlib import contextmanager

_CONTEXT_DEFAULTS = []

@contextmanager
def defaults(**kwargs):
    global _CONTEXT_DEFAULTS
    _CONTEXT_DEFAULTS = [kwargs] + _CONTEXT_DEFAULTS
    yield kwargs
    # print('out defaults', _CONTEXT_DEFAULTS)
    del _CONTEXT_DEFAULTS[-1]

def layers(fun):
    def _wrap(*args, **kwargs):
        parameters = inspect.getfullargspec(fun)[0]
        signature = inspect.signature(fun)
        # print(_CONTEXT_DEFAULTS)
        for parameter in parameters:
            if parameter not in kwargs.keys():
                for ctx in _CONTEXT_DEFAULTS:
                    if parameter in ctx.keys():
                        kwargs[parameter] = ctx[parameter]
                        break
        # print('args:', args)
        # print('kwargs:', kwargs)
        if signature.return_annotation is signature.empty:
            return fun(*args, **kwargs)
        else:
            fun(*args, **kwargs)
    return _wrap
