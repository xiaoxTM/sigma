from sys import platform
import re

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

def clear(string):
    pattern = '\\033\[[0349][0-9]?m'
    splits = re.split(pattern,string)
    return ''.join(splits)

def fill(string):
    stack = []
    pattern = '\\033\[[0349][0-9]?m'
    iters = re.finditer(pattern,string)
    ret = ''
    prev = 0
    for it in iters:
        ret += string[prev:it.end()]
        if it.group(0) == reset:
            stack.pop()
            if len(stack) > 0:
                ret += stack[-1]
        else:
            stack.append(it.group(0))
        prev = it.end()
    ret += reset * len(stack)
    return ret

#if __name__ == '__main__':
#    string = '\033[32mT\033[33ms\033[0mdf\033[0m'
#    print(string)
#    print(clear(string))
#    print(fill(string))
