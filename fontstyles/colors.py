from .base import fontstyle, reset
from . import foreground as fg

@fontstyle
def red(value, fmt=None):
    return '{}{}{}'.format(fg.red, value, reset)

@fontstyle
def green(value, fmt=None):
    return '{}{}{}'.format(fg.green, value, reset)

@fontstyle
def blue(value, fmt=None):
    return '{}{}{}'.format(fg.blue, value, reset)

@fontstyle
def black(value, fmt=None):
    return '{}{}{}'.format(fg.black, value, reset)

@fontstyle
def orange(value, fmt=None):
    return '{}{}{}'.format(fg.orange, value, reset)

@fontstyle
def purple(value, fmt=None):
    return '{}{}{}'.format(fg.purple, value, reset)

@fontstyle
def cyan(value, fmt=None):
    return '{}{}{}'.format(fg.cyan, value, reset)

@fontstyle
def pink(value, fmt=None):
    return '{}{}{}'.format(fg.pink, value, reset)

@fontstyle
def yellow(value, fmt=None):
    return '{}{}{}'.format(fg.yellow, value, reset)
