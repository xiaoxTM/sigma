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

from . import core

def get(optimizer, **kwargs):
    """ get optimizer from None | string | Tensor | callable function
    """
    if optimizer is None:
        return None
    elif isinstance(optimizer, str):
        return core.get_optimizer(optimizer, **kwargs)
    elif isinstance(optimizer, core.Optimizer) or callable(optimizer):
        return optimizer
    else:
        raise ValueError('cannot get optimizer `{}` with type {}'
                         .format(optimizer, type(optimizer)))
