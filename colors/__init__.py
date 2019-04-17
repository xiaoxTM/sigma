"""
    sigma, a deep neural network framework.
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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


def color(fun):
    def _color(value, fmt=None):
        """ fmt should be python .format specification
        """
        if fmt is None:
            fmt = '{}'
        return fun(fmt.format(value), None)
    return _color


@color
def red(value, fmt=None):
    return '{}{}{}'.format(fg.red, value, reset)


@color
def green(value, fmt=None):
    return '{}{}{}'.format(fg.green, value, reset)


@color
def blue(value, fmt=None):
    return '{}{}{}'.format(fg.blue, value, reset)


@color
def black(value, fmt=None):
    return '{}{}{}'.format(fg.black, value, reset)


@color
def orange(value, fmt=None):
    return '{}{}{}'.format(fg.orange, value, reset)


@color
def purple(value, fmt=None):
    return '{}{}{}'.format(fg.purple, value, reset)


@color
def cyan(value, fmt=None):
    return '{}{}{}'.format(fg.cyan, value, reset)


@color
def pink(value, fmt=None):
    return '{}{}{}'.format(fg.pink, value, reset)


@color
def yellow(value, fmt=None):
    return '{}{}{}'.format(fg.yellow, value, reset)


def get():
    return None


def set(config):
    pass
