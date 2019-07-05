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

import numpy as np
from .. import colors, helpers
from . import helper, core

# @helpers.typecheck(input_shape=list,
#                    weights=[list, float],
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def add(inputs_shape,
        weights=None,
        reuse=False,
        name=None,
        scope=None):
    """ add multiple tensors into one
        with weight `weights`
    """
    helper.check_shape_consistency(inputs_shape)
    if weights is None:
        weights = [1] * len(inputs_shape)
    ops_scope, _, _ = helper.assign_scope(name, scope, 'add', reuse)
    def _add(x):
        with ops_scope:
            x = [e * w for e, w in zip(x, weights)]
            return core.add(x)
    return _add, inputs_shape[0]


# @helpers.typecheck(input_shape=list,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def mul(inputs_shape, reuse=False, name=None, scope=None):
    helper.check_shape_consistency(inputs_shape)
    ops_scope, _, _ = helper.assign_scope(name, scope, 'mul', reuse)
    def _mul(x):
        with ops_scope:
            return core.multiply(x[0], x[1])
    return _mul, inputs_shape[0]


# @helpers.typecheck(input_shape=list,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def matmul(inputs_shape, reuse=False, name=None, scope=None):
    helper.check_shape_consistency(inputs_shape)
    ops_scope, _, _ = helper.assign_scope(name, scope, 'matmul', reuse)
    def _matmul(x):
        with ops_scope:
            return core.matmul(x[0], x[1])
    return _matmul, inputs_shape[0]
