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

from .core import trainable_parameters

def get():
    return {'data_format' : core.data_format,
            'epsilon' : core.epsilon
           }


def set(config):
    if config is not None:
        core.epsilon = config.get('epsilon', 1e-5)
        core.data_format = config.get('data_format', 'NHWC')
    else:
        core.epsilon = 1e-5
        core.data_format = 'NHWC'
    core.axis = -1
    if core.data_format == 'NCHW':
        core.axis = 1
