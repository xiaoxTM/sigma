from torch import nn, optim
import typing
from sigma.core.utils.params import parse_params

__acts__ = {
    'elu': nn.ELU,
    'hardshrink': nn.Hardshrink,
    'hardsigmoid': nn.Hardsigmoid,
    'hardtanh': nn.Hardtanh,
    'hardswish': nn.Hardswish,
    'lrelu': nn.LeakyReLU,
    'logsigmoid': nn.LogSigmoid,
    'mhatt': nn.MultiheadAttention,
    'prelu': nn.PReLU,
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'rrelu': nn.RReLU,
    'selu': nn.SELU,
    'celu': nn.CELU,
    'gelu': nn.GELU,
    'sigmoid': nn.Sigmoid,
    'silu': nn.SiLU,
    'mish': nn.Mish,
    'softplus': nn.Softplus,
    'softshrink': nn.Softshrink,
    'softsign': nn.Softsign,
    'tanh': nn.Tanh,
    'tanhshrink': nn.Tanhshrink,
    'glu': nn.GLU,
    'threshold': nn.Threshold,
    'softmin': nn.Softmin,
    'softmax': nn.Softmax,
    'softmax2d': nn.Softmax2d,
    'logsoftmax': nn.LogSoftmax,
    'aapool1d': nn.AdaptiveAvgPool1d,
}


def get(name: str, optimizer: optim.Optimizer):
    assert optimizer is not None and isinstance(optimizer, optim.Optimizer)
    key, params = parse_params(name)
    assert key in __acts__.keys(), f'activations `{key}` not implemented yet'
    return __acts__[key](optimizer, **params)


def register(name: str, activation: typing.Callable):
    assert name.lower() not in __acts__, f'activations `{name.lower()}` already registered'
    __acts__[name.lower()] = activation
