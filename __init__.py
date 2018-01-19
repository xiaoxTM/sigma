from .layers import defaults, _colormaps
from . import status
import os
import os.path
import json
import logging


__version__ = '0.1.0'


def set_print(mode=True):
    if mode is None or isinstance(mode, bool):
        status.graph = mode
    else:
        raise TypeError('mode must be None or True/False, given {}'.format(mode))


config_path = os.path.join(os.environ['HOME'], '.sigma', 'config.json')
if os.path.isfile(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    set_print(config.get('graph', None))
    status.set_data_format(config.get('data_format', 'NHWC'))
    cmap = config.get('colormaps', None)
    if cmap is not None:
        _colormaps.update(cmap)
    logging.debug(config)
else:
    os.makedirs(os.path.join(os.environ['HOME'], '.sigma'), exist_ok=True)
    config = {'graph': None,
              'data_format':'NHWC',
              'colormaps':_colormaps}
    set_print(None)
    status.set_data_format('NHWC')
    with open(config_path, 'w') as f:
        json.dump(config, f)
