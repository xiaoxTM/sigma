import functools
import inspect
from contextlib import contextmanager
from . import utils
from sigma.fontstyles import colors

_CONTEXT_DEFAULTS_={}


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
                if name in ctx_keys:
                    lists = _CONTEXT_DEFAULTS_[name]
                    found = False
                    # find from context traversely
                    for (v,funcs) in lists:
                        if len(funcs) == 0 or signature in funcs:
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
    return kwargs


def defaultable(fun):
    ''' decorate to make function available
        to defaults() as function lists
    '''
    @functools.wraps(fun)
    def _wrap(*args, **kwargs):
        kwargs = assign(fun, *args, **kwargs)
        return fun(**kwargs)
    return _wrap


@contextmanager
def defaults(*args, **kwargs):
    ''' set defaults to context
        Attrinutes
        ==========
            args: callable function list
            kwargs: dict of key-value pairs
    '''
    for idx, arg in enumerate(args):
        if not callable(arg):
            raise TypeError('args at {}-th is not callable. given: {}'.format(idx, arg))

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


def merge_args(args, kwargs, func):
    wargs = {}
    if callable(func):
        sig = inspect.signature(func)
        names = list(sig.parameters.keys())
    elif isinstance(func, (list,tuple)):
        names = func
    else:
        raise TypeError('func must be function or list/tuple, given {}'.format(type(func)))
    for i, arg in enumerate(args):
        wargs.update({names[i]:arg})
    wargs.update(kwargs)
    return wargs


def split_params(option:str):
    ''' split parameters by ;
        e.g., given option=[cce(reduce=mean,weight=0.2);margin(pos=0.9,neg=0.1)]
              produce:
                  ['cce(reduce=mean,weight=0.2)', 'margin(pos=0.9,neg=0.1)']
    '''
    options = option.strip().split(';')
    if len(options) == 1:
        return options
    options[0] = options[0][1:] # get rid of `[`
    options[-1] = options[-1][:-1] # get rid of `]`
    return options

@utils.deprecated('{} is deprecated, please use {} instead'.format(colors.red('parse_parameters'), colors.blue('parse_params')))
def parse_parameters(option:str):
    ''' parse parameters from option
        option should have form like:
            'multistep<milestones:[10,20]|gamma:0.1|key:"value">'
    '''
    options = option.strip().split('<')
    if len(options) == 1:
        return options[0], {}
    options[1] = options[1][:-1] # remove '>'
    params = options[1].strip().split('|')
    parameters = dict()

    for param in params:
        k, v = param.strip().split(':',1)
        parameters[k] = eval(v)
    return options[0], parameters

def parse_params(option:str):
    '''parse parameters from option
       string should have form like:
           'multistep(milestones=[10,20],gamma=0.1)'
       return multistep:option
              {milestones:[10,20],
               gamma:0.1}:dict
    '''
    options = option.strip().rsplit('(',1)
    if len(options) == 1:
        return options[0], {}
    options[1] = options[1][:-1] # remove ')'
    params = options[1].strip().split(',')
    parameters = dict()

    for param in params:
        k, v = param.strip().split('=',1)
        parameters[k] = eval(v)
    return options[0], parameters

def expand_params(params, size):
    ''' expand params to list of list
    '''
    if not isinstance(params, (list, tuple)):
        params = [params]
    expanded = [None] * len(params)
    for idx, param in enumerate(params):
        if not isinstance(param, (list, tuple)):
            param = [param]
        expanded[idx] = param + [param[-1]]*(size - len(param))
    return expanded

def expand_param(param, size):
    ''' expand params to list
    '''
    if not isinstance(param, (list, tuple)):
        param = [param]
    return param + [param[-1]] * (size-len(param))