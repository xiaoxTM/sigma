from sys import platform

reset = '\033[0m'

if platform.startswith('win'):
    reset = ''

def fontstyle(fun):
    def _fontstyle(value, fmt=None):
        """ fmt should be python .format specification
        """
        if fmt is None:
            fmt = '{}'
        return fun(fmt.format(value), None)
    return _fontstyle
