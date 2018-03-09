from .core import predict, session, line, run, train, build
from ..ops.core import __backend__
from ..layers.core import set_print, export_graph

def get():
    return {'backend':__backend__}


def set(config):
    __backend__ = config.get('backend',
                         'tensorflow')
