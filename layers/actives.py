"""
    sigma, a deep neural network framework.
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from .. import ops
from . import core


""" squash function to re-range value to
    [0, 1)
"""
@core.layer
def squash(inputs,
           axis=-2,
           epsilon=ops.core.epsilon,
           safe=True,
           reuse=False,
           name=None,
<<<<<<< HEAD
           scope=None,
           **kwargs):
    return ops.actives.squash(axis, epsilon, safe, True, reuse, name, scope, **kwargs)(inputs)


@core.layer
def crelu(inputs, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.crelu(True, reuse, name, scope, **kwargs)(inputs)
=======
           scope=None):
    fun = ops.actives.squash(axis, epsilon, safe, True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)


@core.layer
def crelu(inputs, reuse=False, name=None, scope=None):
    fun = ops.actives.crelu(True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" relu activates
    calculates:
        max(x, 0)
"""
@core.layer
<<<<<<< HEAD
def relu(inputs, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.relu(True, reuse, name, scope, **kwargs)(inputs)
=======
def relu(inputs, reuse=False, name=None, scope=None):
    fun = ops.actives.relu(True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" relu6 activates
    calculates:
        min(max(x, 0), 6)
"""
@core.layer
<<<<<<< HEAD
def relu6(inputs, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.relu6(True, reuse, name, scope, **kwargs)(inputs)
=======
def relu6(inputs, reuse=False, name=None, scope=None):
    fun = ops.actives.relu6(True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" elu activates
    calculates:
        exp(x) -1 if x < 0 else x
"""
@core.layer
<<<<<<< HEAD
def elu(inputs, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.elu(True, reuse, name, scope, **kwargs)(inputs)
=======
def elu(inputs, reuse=False, name=None, scope=None):
    fun = ops.actives.elu(True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


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
<<<<<<< HEAD
         scope=None,
         **kwargs):
    return ops.actives.selu(alpha, scale, True, reuse, name, scope, **kwargs)(inputs)
=======
         scope=None):
    fun = ops.actives.selu(alpha, scale, True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" leaky_relu activates
    calculates:
        max(x, alpha*x)
"""
@core.layer
<<<<<<< HEAD
def leaky_relu(inputs, alpha=0.2, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.leaky_relu(alpha, True, reuse, name, scope, **kwargs)(inputs)
=======
def leaky_relu(inputs, alpha=0.2, reuse=False, name=None, scope=None):
    fun = ops.actives.leaky_relu(alpha, True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" softmax activates
    calculates:
        exp(x) / reduce_sum(exp(x), dim)
"""
@core.layer
<<<<<<< HEAD
def softmax(inputs, dim=-1, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.softmax(dim, True, reuse, name, scope, **kwargs)(inputs)
=======
def softmax(inputs, dim=-1, reuse=False, name=None, scope=None):
    fun = ops.actives.softmax(dim, True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" softplus activates
    calculates:
        log(exp(x) + 1)
"""
@core.layer
<<<<<<< HEAD
def softplus(inputs, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.softplus(True, reuse, name, scope, **kwargs)(inputs)
=======
def softplus(inputs, reuse=False, name=None, scope=None):
    fun = ops.actives.softplus(True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" softsign activates
    calculates:
        x / (abs(x) + 1)
"""
@core.layer
<<<<<<< HEAD
def softsign(inputs, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.softsign(True, reuse, name, scope, **kwargs)(inputs)
=======
def softsign(inputs, reuse=False, name=None, scope=None):
    fun = ops.actives.softsign(True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" sigmoid activates
    calculates:
        1 / (1 + exp(-x))
"""
@core.layer
<<<<<<< HEAD
def sigmoid(inputs, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.sigmoid(True, reuse, name, scope, **kwargs)(inputs)
=======
def sigmoid(inputs, reuse=False, name=None, scope=None):
    fun = ops.actives.sigmoid(True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" hyperbolic tangent activates
    calculates:
        (exp(x) -1) / (exp(x) + 1)
"""
@core.layer
<<<<<<< HEAD
def tanh(inputs, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.tanh(True, reuse, name, scope, **kwargs)(inputs)
=======
def tanh(inputs, reuse=False, name=None, scope=None):
    fun = ops.actives.tanh(True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a


""" linear activates
    calculates:
        x
"""
@core.layer
<<<<<<< HEAD
def linear(inputs, reuse=False, name=None, scope=None, **kwargs):
    return ops.actives.linear(True, reuse, name, scope, **kwargs)(inputs)
=======
def linear(inputs, reuse=False, name=None, scope=None):
    fun = ops.actives.linear(True, reuse, name, scope)
    return core.run_and_record_fun(fun, name, inputs)
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a
