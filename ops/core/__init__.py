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

from sigma import colors
from .commons import *

__backend__ = 'tensorflow'

if __backend__ == 'tensorflow':
    from .__tensorflow__ import *
elif __backend__ == 'theano':
    from .__theano__ import *
elif __backend__ == 'pytorch':
    from .__pytorch__ import *
else:
    raise ValueError('`{}` backend for sigma is not supported'
                     .format(__backend__))
print('Using {} {}<{}>{} backend'.format(colors.red(__backend__),
                                       colors.fg.green,
                                       version,
                                       colors.reset))
