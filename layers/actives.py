from .. import ops
from . import core

@core.layer
def crelu(inputs, reuse=False, name=None, scope=None):
    return ops.actives.crelu(reuse, name, scope)(inputs)


""" relu activates
    calculates:
        max(x, 0)
"""
@core.layer
def relu(inputs, reuse=False, name=None, scope=None):
    return ops.actives.relu(reuse, name, scope)(inputs)


""" relu6 activates
    calculates:
        min(max(x, 0), 6)
"""
@core.layer
def relu6(inputs, reuse=False, name=None, scope=None):
    return ops.actives.relu6(reuse, name, scope)(inputs)


""" elu activates
    calculates:
        exp(x) -1 if x < 0 else x
"""
@core.layer
def elu(inputs, reuse=False, name=None, scope=None):
    return ops.actives.elu(reuse, name, scope)(inputs)


""" elu activates
    calculates:
        scale * alpha * (exp(x) -1)
"""
@core.layer
def selu(inputs,
         alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946,
         reuse=False,
         name=None,
         scope=None):
    return ops.actives.selu(alpha, scale, reuse, name, scope)(inputs)


""" leaky_relu activates
    calculates:
        max(x, alpha*x)
"""
@core.layer
def leaky_relu(inputs, alpha=0.2, reuse=False, name=None, scope=None):
    return ops.actives.leaky_relu(alpha, reuse, name, scope)(inputs)


""" softmax activates
    calculates:
        exp(x) / reduce_sum(exp(x), dim)
"""
@core.layer
def softmax(inputs, dim=-1, reuse=False, name=None, scope=None):
    return ops.actives.softmax(dim, reuse, name, scope)(inputs)


""" softplus activates
    calculates:
        log(exp(x) + 1)
"""
@core.layer
def softplus(inputs, reuse=False, name=None, scope=None):
    return ops.actives.softplus(reuse, name, scope)(inputs)


""" softsign activates
    calculates:
        x / (abs(x) + 1)
"""
@core.layer
def softsign(inputs, reuse=False, name=None, scope=None):
    return ops.actives.softsign(reuse, name, scope)(inputs)


""" sigmoid activates
    calculates:
        1 / (1 + exp(-x))
"""
@core.layer
def sigmoid(inputs, reuse=False, name=None, scope=None):
    return ops.actives.sigmoid(reuse, name, scope)(inputs)


""" hyperbolic tangent activates
    calculates:
        (exp(x) -1) / (exp(x) + 1)
"""
@core.layer
def tanh(inputs, reuse=False, name=None, scope=None):
    return ops.actives.tanh(reuse, name, scope)(inputs)


""" linear activates
    calculates:
        x
"""
@core.layer
def linear(inputs, reuse=False, name=None, scope=None):
    return ops.actives.linear(reuse, name, scope)(inputs)
