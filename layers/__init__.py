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

from . import convolutional as convs
from . import normalization as norms
from . import regularizers as regus
# from . import optimizers as opts
from . import math
from . import actives
from . import pools
from . import base
from . import merge
from . import losses
from . import metrics
from . import core
from . import capsules
from .core import defaults


def get():
    return {'graph' : core.__graph__,
            'details' : core.__details__,
            'defaults' : core.__defaults__,
            'colormaps' : core.__colormaps__
            }


def set(config):
    if config is not None:
        core.__graph__ = config.get('graph', False)
        core.__details__ = config.get('details', False)
        value = config.get('defaults', None)
        if value is not None:
            core.__defaults__ = value
        value = config.get('colormaps', None)
        if value is not None:
            core.__colormaps____ = value
