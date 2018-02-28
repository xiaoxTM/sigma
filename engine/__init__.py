from .core import [predict, session, line, run, train]
from .core import __backend__ as backend


def get():
    return backend


def set(config):
    backend = config.get('backend',
                         'tensorflow')
