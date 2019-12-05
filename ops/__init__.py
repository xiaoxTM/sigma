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
from . import convolutional as convs
from . import normalization as norms
from . import regularizers as regus
from . import optimizers as opts
from . import math
from . import capsules
from . import actives
from . import pools
from . import merge
from . import helper
from . import base
from . import losses
from . import metrics

from .core import trainable_parameters, version_compare_great

def str2type(name):
    return eval('core.{}'.format(name))


def type2str(dtype):
    if dtype == core.float16:
        return 'float16'
    elif dtype == core.float32:
        return 'float32'
    elif dtype == core.float64:
        return 'float64'
    else:
        raise ValueError('dtype `{}` not supported.'.format(dtype))

def get():
    return {'data_format' : core.data_format,
                    'epsilon': core.epsilon,
                    'floatx': type2str(core.floatx)}


def set(config):
    if config is not None:
        if version_compare_great(core.version, '1.5.0'):
            core.data_format = config.get('data_format', ['NC', 'NWC', 'NHWC', 'NDHWC'])
        else:
            core.data_format = config.get('data_format', ['NHWC', 'NHWC', 'NHWC', 'NHWC'])
        core.epsilon = config.get('epsilon', 1e-9)
        core.floatx = str2type(config.get('floatx', 'float32'))
    else:
        if version_compare_great(core.version, '1.5.0'):
            core.data_format = ['NC', 'NWC', 'NHWC', 'NDHWC']
        else:
            core.data_format = config.get('data_format', ['NHWC', 'NHWC', 'NHWC', 'NHWC'])
        core.epsilon = 1e-9
        core.floatx = core.float32
    core.caxis = -1
    if core.data_format[1] == 'NCW':
        core.caxis = 1
