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

from . import status, ops, layers, helpers, engine, dbs, colors
from .layers import defaults
from .ops.core import placeholder, seed
from .engine import session, predict, build_experiment
import os
import os.path
import json
import logging


__version__ = '0.1.3.4'


__packages__ = {'sigma.status' : status,
                'sigma.ops' : ops,
                'sigma.layers' : layers,
                'sigma.helpers' : helpers,
                'sigma.engine' : engine,
                'sigma.dbs' : dbs,
                'sigma.colors' : colors
               }


config_path = os.path.join(os.environ['HOME'], '.sigma', 'config.json')
if os.path.isfile(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key, package in __packages__.items():
        package.set(config.get(key, {}))
    logging.debug(config)
else:
    os.makedirs(os.path.join(os.environ['HOME'], '.sigma'), exist_ok=True)
    config = {}
    for key, package in __packages__.items():
        config[key] = package.get()
    logging.debug(config)
    with open(config_path, 'w') as f:
        json.dump(config, f)
