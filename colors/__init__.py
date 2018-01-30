from sys import platform
from . import foreground as fg
from . import background as bg
reset = '\033[0m'
bold = '\033[01m'
disable = '\033[02m'
underline = '\033[04m'
reverse = '\033[07m'
invisible = '\033[08m'
strikethrough = '\033[09m'

if platform.startswith('win'):
    reset = ''
    bold = ''
    disable = ''
    underline = ''
    reverse = ''
    invisible = ''
    strikethrough = ''


def red(string):
    return '{}{}{}'.format(fg.red, string, reset)


def green(string):
    return '{}{}{}'.format(fg.green, string, reset)


def blue(string):
    return '{}{}{}'.format(fg.blue, string, reset)


def black(string):
    return '{}{}{}'.format(fg.black, string, reset)


def orange(string):
    return '{}{}{}'.format(fg.orange, string, reset)


def purple(string):
    return '{}{}{}'.format(fg.purple, string, reset)


def cyan(string):
    return '{}{}{}'.format(fg.cyan, string, reset)


def pink(string):
    return '{}{}{}'.format(fg.pink, string, reset)


def yellow(string):
    return '{}{}{}'.format(fg.yellow, string, reset)


def get():
    return None


def set(config):
    pass
