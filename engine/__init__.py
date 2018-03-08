from .core import predict, session, line, run, train
from ..ops.core import __backend__

def get():
    return {'backend':__backend__}


def set(config):
    __backend__ = config.get('backend',
                         'tensorflow')
