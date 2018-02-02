from . import convolutional as convs
from . import actives
from . import pools
from . import helper
from . import base
from . import losses
from . import core

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
