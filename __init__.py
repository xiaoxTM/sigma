from .layers import defaults
from . import layers

def set_print(mode=True):
    if mode is None or isinstance(mode, bool):
        layers.graph = mode
    else:
        raise TypeError('mode must be None or True/False, given {}'.format(mode))

__version__ = '0.1.0'
