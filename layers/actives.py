from ..ops import actives

def crelu(x, name=None):
    return actives.crelu(name)(x)

""" relu activates
    calculates:
        max(x, 0)
"""
def relu(x, name=None):
    return actives.relu(name)(x)


""" relu6 activates
    calculates:
        min(max(x, 0), 6)
"""
def relu6(x, name=None):
    return actives.relu6(name)(x)


""" elu activates
    calculates:
        exp(x) -1 if x < 0 else x
"""
def elu(x, name=None):
    return actives.elu(name)(x)


""" elu activates
    calculates:
        scale * alpha * (exp(x) -1)
"""
def selu(x, alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946,
         name=None):
    return actives.selu(alpha, scale, name)(x)


""" leaky_relu activates
    calculates:
        max(x, alpha*x)
"""
def leaky_relu(x, alpha=0.2, name=None):
    return actives.leaky_relu(alpha, name)(x)


""" softmax activates
    calculates:
        exp(x) / reduce_sum(exp(x), dim)
"""
def softmax(x, dim=-1, name=None):
    return actives.softmax(dim, name)(x)


""" softplus activates
    calculates:
        log(exp(x) + 1)
"""
def softplus(x, name=None):
    return actives.softplus(name)(x)


""" softsign activates
    calculates:
        x / (abs(x) + 1)
"""
def softsign(x, name=None):
    return actives.softsign(name)(x)


""" sigmoid activates
    calculates:
        1 / (1 + exp(-x))
"""
def sigmoid(x, name=None):
    return actives.sigmoid(name)(x)


""" linear activates
    calculates:
        x
"""
def linear(x):
    return actives.linear(x)
