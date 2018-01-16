from ..ops import actives, helper
from .layers import layers

@layers
def crelu(inputs, reuse=False, name=None):
    return actives.crelu(name)(inputs)


""" relu activates
    calculates:
        max(x, 0)
"""
@layers
def relu(inputs, reuse=False, name=None):
    return actives.relu(name)(inputs)


""" relu6 activates
    calculates:
        min(max(x, 0), 6)
"""
@layers
def relu6(inputs, reuse=False, name=None):
    return actives.relu6(name)(inputs)


""" elu activates
    calculates:
        exp(x) -1 if x < 0 else x
"""
@layers
def elu(inputs, reuse=False, name=None):
    return actives.elu(name)(inputs)


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
    return actives.selu(alpha, scale, name)(inputs)


""" leaky_relu activates
    calculates:
        max(x, alpha*x)
"""
@layers
def leaky_relu(inputs, alpha=0.2, reuse=False, name=None):
    return actives.leaky_relu(alpha, name)(inputs)


""" softmax activates
    calculates:
        exp(x) / reduce_sum(exp(x), dim)
"""
@layers
def softmax(inputs, dim=-1, reuse=False, name=None):
    return actives.softmax(dim, name)(inputs)


""" softplus activates
    calculates:
        log(exp(x) + 1)
"""
@layers
def softplus(inputs, reuse=False, name=None):
    return actives.softplus(name)(inputs)


""" softsign activates
    calculates:
        x / (abs(x) + 1)
"""
@layers
def softsign(inputs, reuse=False, name=None):
    return actives.softsign(name)(inputs)


""" sigmoid activates
    calculates:
        1 / (1 + exp(-x))
"""
@layers
def sigmoid(inputs, reuse=False, name=None):
    return actives.sigmoid(name)(inputs)


""" hyperbolic tangent activates
    calculates:
        (exp(x) -1) / (exp(x) + 1)
"""
@layers
def tanh(inputs, reuse=False, name=None):
    return actives.tanh(name)(inputs)


""" linear activates
    calculates:
        x
"""
@layers
def linear(inputs, reuse=False, name=None):
    return actives.linear(name)(inputs)
