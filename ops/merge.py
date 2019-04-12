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
#                    axis=int,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def concat(inputs_shape,
           axis=-1,
           reuse=False,
           name=None,
           scope=None):
    """ concatenate multiple tensors into one
        along with axis
    """
    if not isinstance(inputs_shape, (list, tuple)):
        raise TypeError('concat requires inputs as '
                        '{}list / tpule{}, given {}'
                        .format(colors.fg.green, colors.reset,
                                colors.red(type(inputs_shape))))
    elif len(inputs_shape) < 2:
        raise ValueError('concat requires at least {}two{} inputs, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(len(inputs_shape))))
    output_shape = inputs_shape[0]
    helper.check_input_shape(output_shape)
    for idx, ip in enumerate(inputs_shape[1:]):
        helper.check_input_shape(ip)
        if not np.all(np.delete(output_shape, axis) == np.delete(ip, axis)):
            raise ValueError('shape of {}-input differs from first'
                             ' one besides {}-axis. {} vs {}'
                             .format(colors.red(idx+1),
                                     colors.green(axis),
                                     colors.red(output_shape),
                                     colors.green(ip)))
        output_shape[axis] += ip[axis]
    ops_scope, _, _ = helper.assign_scope(name, scope, 'concat', reuse)
    def _concat(x):
        with ops_scope:
            return core.concat(x, axis)
    return _concat, output_shape
