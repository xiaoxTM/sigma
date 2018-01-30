from . import status, ops, layers, helpers, engine, dbs, colors
from .layers import defaults
from .ops import placeholder
from .engine import run, session, predict
import os
import os.path
import json
import logging


__version__ = '0.1.2'


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
        package.set(config.get(key, None))
    logging.debug(config)
else:
    os.makedirs(os.path.join(os.environ['HOME'], '.sigma'), exist_ok=True)
    config = {}
    for key, package in __packages__:
        config[key] = package.get()
    logging.debug(config)
    with open(config_path, 'w') as f:
        json.dump(config, f)
