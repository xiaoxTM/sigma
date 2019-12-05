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

@core.layer
def total_variation_regularize(inputs,
                               reuse=False,
                               name=None,
                               scope=None):
    input_shape = ops.helper.norm_input_shape(inputs)
    return ops.regularizers.total_variation_regularizer(input_shape,
                                                        reuse,
                                                        name,
                                                        scope)(inputs)

tvr = total_variation_regularize
