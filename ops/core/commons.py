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

# 'NC' for fully connected
# 'NWC' for 1d
# 'NHWC' for 2d
data_format = ['NC', 'NWC', 'NHWC', 'NDHWC']
caxis = -1 # channel axis

def shape_statistics(shape):
    """ statistics shapes in shape_list
        that how many Nones and -1s
        For example:
        > shape_list [None, 1, 3, 2, 4, -1, None]
        > stats = {'None':[0, 6], '-1':[5]}
    """
    stats = {'nones': [], '-1': []}
    for idx, s in enumerate(shape):
        if s is None:
            stats['nones'].append(idx)
        elif s == -1:
            stats['-1'].append(idx)
    return stats


def encode(strings, codec='utf8'):
    if isinstance(strings, str):
        return strings.encode(codec)
    elif isinstance(strings, (list, tuple, np.ndarray)):
        def _encode(string):
            if isinstance(string, str):
                return string.encode(codec)
            elif isinstance(string, np.bytes_):
                return string.tostring().encode(codec)
            else:
                raise TypeError('`string` must be string, given {}'
                                .format(type(string)))
        return list(map(_encode, strings))
    else:
        raise TypeError('`strings` must be string or '
                        'list/tuple of string, given {}'
                        .format(type(strings)))


def decode(strings, codec='utf8'):
    if isinstance(strings, str):
        return strings.decode(codec)
    elif isinstance(strings, (list, tuple, np.ndarray)):
        def _decode(string):
            if isinstance(string, str):
                return string.decode(codec)
            elif isinstance(string, np.bytes_):
                return string.tostring().decode(codec)
            else:
                raise TypeError('`string` must be string, given {}'
                                .format(type(string)))
        return list(map(_decode, strings))
    else:
        raise TypeError('`strings` must be string or '
                        'list/tuple of string, given {}'
                        .format(type(strings)))
