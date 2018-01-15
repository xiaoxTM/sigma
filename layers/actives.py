from ..ops import actives, helper

from . import layers

def _actives(fun, inputs, typename, reuse, name):
    x = fun(inputs)
    helper.print_layer(inputs, x, typename, reuse, name)
    return x


@layers
def crelu(inputs, reuse=False, name=None):
    fun = actives.crelu(name)
    return _actives(fun, inputs, 'crelu', reuse, name)


""" relu activates
    calculates:
        max(x, 0)
"""
@layers
def relu(inputs, reuse=False, name=None):
    fun = actives.relu(name)
    return _actives(fun, inputs, 'relu', reuse, name)


""" relu6 activates
    calculates:
        min(max(x, 0), 6)
"""
@layers
def relu6(inputs, reuse=False, name=None):
    fun = actives.relu6(name)
    return _actives(fun, inputs, 'relu6', reuse, name)


""" elu activates
    calculates:
        exp(x) -1 if x < 0 else x
"""
@layers
def elu(inputs, reuse=False, name=None):
    fun = actives.elu(name)
    return _actives(fun, inputs, 'elu', reuse, name)


""" elu activates
    calculates:
        scale * alpha * (exp(x) -1)
"""
@layers
def selu(inputs,
         alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946,
         reuse=False,
         name=None):
    fun = actives.selu(alpha, scale, name)
    return _actives(fun, inputs, 'selu', reuse, name)


""" leaky_relu activates
    calculates:
        max(x, alpha*x)
"""
@layers
def leaky_relu(inputs, alpha=0.2, reuse=False, name=None):
    fun = actives.leaky_relu(alpha, name)
    return _actives(fun, inputs, 'leaky_relu', reuse, name)


""" softmax activates
    calculates:
        exp(x) / reduce_sum(exp(x), dim)
"""
@layers
def softmax(inputs, dim=-1, reuse=False, name=None):
    fun = actives.softmax(dim, name)
    return _actives(fun, inputs, 'softmax', reuse, name)


""" softplus activates
    calculates:
        log(exp(x) + 1)
"""
@layers
def softplus(inputs, reuse=False, name=None):
    fun = actives.softplus(name)
    return _actives(fun, inputs, 'softplus', reuse, name)


""" softsign activates
    calculates:
        x / (abs(x) + 1)
"""
@layers
def softsign(inputs, reuse=False, name=None):
    fun = actives.softsign(name)
    return _actives(fun, inputs, 'softsign', reuse, name)


""" sigmoid activates
    calculates:
        1 / (1 + exp(-x))
"""
@layers
def sigmoid(inputs, reuse=False, name=None):
    fun = actives.sigmoid(name)
    return _actives(fun, inputs, 'sigmoid', reuse, name)


""" hyperbolic tangent activates
    calculates:
        (exp(x) -1) / (exp(x) + 1)
"""
@layers
def tanh(inputs, reuse=False, name=None):
    fun = actives.tanh(name)
    return _actives(fun, inputs, 'tanh', reuse, name)


""" linear activates
    calculates:
        x
"""
@layers
def linear(inputs, reuse=False, name=None):
    fun = actives.linear(name)
    return _actives(fun, inputs, 'linear', reuse, name)
