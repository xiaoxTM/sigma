from jittor import nn
from sigma import parse_params
from sigma.fontstyles import colors
import sigma.version as svn

__acts__ = {'elu':nn.ELU,
            'lrelu':nn.LeakyReLU,
            'leakyrelu':nn.LeakyReLU,
            'prelu':nn.PReLU,
            'relu':nn.ReLU,
            'relu6':nn.ReLU6,
            'sig':nn.Sigmoid,
            'sigmoid':nn.Sigmoid,
            'softplus':nn.Softplus,
            'tanh':nn.Tanh,
            'softmax':nn.Softmax}

def get(activation):
    if activation is None or isinstance(activation, nn.Module) or callable(activation):
        return activation
    elif isinstance(activation, str):
        act_type, params = parse_params(activation)
        act_type = act_type.lower()
        assert act_type in __acts__.keys(), 'activation type {} not support'.format(act_type)
        return __acts__[act_type](**params)
    else:
        raise TypeError('cannot conver type {} into Activation'.format(colors.red(type(activation))))

def register(key, activation):
    assert key is not None and activation is not None, 'both key and activation can not be none'
    global __acts__
    assert key not in __acts__.keys(), 'key {} already registered'.format(key)
    assert isinstance(activation, nn.Module) or callable(activation), 'activation must be either an instance of nn.Module or function, given {}'.format(type(activation))
    __acts__.update({key:activation})
