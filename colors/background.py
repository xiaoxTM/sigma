from sys import platform

black = '\033[40m'
red = '\033[41m'
green = '\033[42m'
orange = '\033[43m'
blue = '\033[44m'
purple = '\033[45m'
cyan = '\033[46m'
lightgray = '\033[47m'

if platform.startswith('win'):
    black = ''
    red = ''
    green = ''
    orange = ''
    blue = ''
    purple = ''
    cyan = ''
    lightgray = ''
