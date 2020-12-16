from sys import platform

bold = '\033[01m'
disable = '\033[02m'
italic = '\033[03m'
underline = '\033[04m'
blink = '\033[05m'
blink2 = '\033[06m'
reverse = '\033[07m'
invisible = '\033[08m'
strikethrough = '\033[09m'

if platform.startswith('win'):
    bold = ''
    disable = ''
    italic = ''
    underline = ''
    blink = ''
    blink2 = ''
    reverse = ''
    invisible = ''
    strikethrough = ''
