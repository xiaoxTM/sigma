from .layers import defaults
from . import layers
import os
import os.path
import json

def set_print(mode=True):
    if mode is None or isinstance(mode, bool):
        layers._graph = mode
    else:
        raise TypeError('mode must be None or True/False, given {}'.format(mode))

__version__ = '0.1.0'


config_path = os.path.join(os.environ['HOME'], '.sigma', 'config.json')
if os.path.isfile(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    set_print(config['graph'])
else:
    os.makedirs(os.path.join(os.environ['HOME'], '.sigma'), exist_ok=True)
    config = {'graph': None}
    set_print(None)
    with open(config_path, 'w') as f:
        json.dump(config, f)
