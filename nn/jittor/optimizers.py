from jittor.optim import *
from sigma import parse_params,  version
from sigma.fontstyles import colors

__optimizers__ = {'adam': Adam,
                  'rmsprop': RMSprop,
                  'sgd': SGD}

#if version.ge(torch.__version__, '1.2.0'):
#    __optimizers__.update({'adamw': AdamW})

def get(optimizer, lr, parameters):
    if optimizer is None or isinstance(optimizer, Optimizer):
        return optimizer
    elif isinstance(optimizer, str):
        optimizer_type, params = parse_params(optimizer)
        optimizer_type = optimizer_type.lower()
        assert optimizer_type in __optimizers__.keys(), 'optimizer type {} not support'.format(optimizer_type)
        return __optimizers__[optimizer_type](parameters, lr, **params)
    else:
        raise TypeError('cannot convert type {} into Optimizer'.format(colors.red(type(optimizer))))

def register(key, optimizer):
    assert key is not None and optimizer is not None, 'both key and optimizer can not be none'
    global __optimizers__
    assert key not in __optimizers__.keys(), 'key {} already registered'.format(key)
    assert isinstance(optimizer, Optimizer), 'optimizer must be an instance of Optimizer, given {}'.format(type(optimizer))
    __optimizers__.update({key:optimizer})
