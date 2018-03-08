from . import convolutional as convs
from . import normalization as norms
from . import actives
from . import pools
from . import base
from . import merge
from . import losses
from . import core
from . import capsules
from .core import defaults, export_graph


def get():
    return {'graph' : core.__graph__,
            'defaults' : core.__defaults__,
            'colormaps' : core.__colormaps__
            }


def set(config):
    if config is not None:
        core.__graph__ = config.get('graph', False)
        value = config.get('defaults', None)
        if value is not None:
            core.__defaults__ = value
        value = config.get('colormaps', None)
        if value is not None:
            core.__colormaps____ = value
