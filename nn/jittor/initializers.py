from jittor import init
from sigma import parse_params
from sigma.fontstyles import colors
import functools

__inits__ = {'uniform': init.uniform_,
             'normal': init.gauss_,
             'constant': init.constant_,
             'const': init.constant_,
             'eyes': init.eye_,
             'xavier-uniform': init.xavier_uniform_,
             'xu': init.xavier_uniform_,
             'xavier-normal': init.xavier_gauss_,
             'xn': init.xavier_gauss_,
             'kaiming-uniform': init.kaiming_uniform_,
             'ku': init.kaiming_uniform_,
             'kaiming-normal': init.kaiming_normal_,
             'kn': init.kaiming_normal_}


def get(initializer):
    if  initializer is None or callable(initializer):
        return initializer
    elif isinstance(initializer, str):
        initializer = initializer.strip()
        if initializer in ['','null','none']:
            return None
        initializer_type, params = parse_params(initializer)
        initializer_type = initializer_type.lower()
        assert initializer_type in __inits__.keys(), 'initializer type {} not support'.format(initializer_type)
        return functools.partial(__inits__[initializer_type], **params)
    raise TypeError('cannot convert type {} into initializer'.format(colors.red(type(initializer))))


def register(key, initializer):
    assert key is not None and initializer is not None, 'both key and initializer can not be none'
    global __inits__
    assert key not in __inits__.keys(), 'key {} alread registered'.format(key)
    assert callable(initializer), 'initializer must be function, given {}'.format(type(initializer))
    __inits__.update({key:initializer})
