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
__version__ = '1.0.0'

import os
import os.path
import json
import logging

from .utils import *
from .fontstyles import *
from sigma import nn

epsilon = 1e-9

config_path = os.path.join(os.environ['HOME'], '.sigma', 'config.json')
if os.path.isfile(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    logging_level = config.get('logging-level',logging.INFO)
    logging_format = config.get('logging-format', '%(asctime)s %(filename)s@%(lineno)s %(levelname)s %(message)s')
    logging_datefmt = config.get('logging-date-format','%Y-%m-%d %H:%M:%S %a')
    logging.basicConfig(level=logging_level,
                        format=logging_format,
                        datefmt=logging_datefmt)
    epsilon = config.get('epsilon', 1e-9)
    conf = config.get('nn', None)
    if conf is not None:
        nn.set(conf)
    logging.debug(config)
else:
    os.makedirs(os.path.join(os.environ['HOME'], '.sigma'), exist_ok=True)
    config = {}
    conf = nn.get()
    if conf is not None:
        config['nn'] = conf
    config['logging-level'] = logging.WARNING
    config['logging-format'] = '%(asctime)s %(filename)s@%(lineno)s %(levelname)s %(message)s'
    config['logging-date-format'] = '%Y-%m-%d %H:%M:%S %a'
    config['epsilon'] = epsilon
    logging.debug(config)
    with open(config_path, 'w') as f:
        json.dump(config, f)
