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
