from .base import fontstyle,reset
from . import fontattrib as fa

@fontstyle
def bold(value, fmt=None):
    return '{}{}{}'.format(fa.bold, value, reset)

@fontstyle
def disable(value, fmt=None):
    return '{}{}{}'.format(fa.disable, value, reset)

@fontstyle
def italic(value, fmt=None):
    return '{}{}{}'.format(fa.italic, value, reset)

@fontstyle
def underline(value, fmt=None):
    return '{}{}{}'.format(fa.underline, value, reset)

@fontstyle
def blink(value, fmt=None):
    return '{}{}{}'.format(fa.blink, value, reset)

@fontstyle
def reverse(value, fmt=None):
    return '{}{}{}'.format(fa.reverse, value, reset)
