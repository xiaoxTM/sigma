import numpy as np
import inspect
from sigma.fontstyles import colors
import functools
import warnings
from datetime import datetime
import sys


def deprecated(message, stacklevel=1, source=None):
    def wrap(func):
        @functools.wraps(func)
        def _deprecated(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel, source)
            return func(*args, **kwargs)
        return _deprecated
    return wrap


def shape_statistics(shape):
    """ statistics shapes in shape_list
        that how many Nones and -1s
        For example:
        > shape_list [None, 1, 3, 2, 4, -1, None]
        > stats = {'None':[0, 6], '-1':[5]}
    """
    stats = {'None': [], '-1': []}
    for idx, s in enumerate(shape):
        if s is None:
            stats['None'].append(idx)
        elif s == -1:
            stats['-1'].append(idx)
    return stats


def encode(strings, codec='utf8'):
    if isinstance(strings, str):
        return strings.encode(codec)
    elif isinstance(strings, (list, tuple, np.ndarray)):
        def _encode(string):
            if isinstance(string, str):
                return string.encode(codec)
            elif isinstance(string, np.bytes_):
                return string.tostring().encode(codec)
            else:
                raise TypeError('`string` must be string, given {}'
                                .format(type(string)))
        return list(map(_encode, strings))
    else:
        raise TypeError('`strings` must be string or '
                        'list/tuple of string, given {}'
                        .format(type(strings)))


def decode(strings, codec='utf8'):
    if isinstance(strings, str):
        return strings.decode(codec)
    elif isinstance(strings, (list, tuple, np.ndarray)):
        def _decode(string):
            if isinstance(string, str):
                return string.decode(codec)
            elif isinstance(string, np.bytes_):
                return string.tostring().decode(codec)
            else:
                raise TypeError('`string` must be string, given {}'
                                .format(type(string)))
        return list(map(_decode, strings))
    else:
        raise TypeError('`strings` must be string or '
                        'list/tuple of string, given {}'
                        .format(type(strings)))


def intsize(x, cminus=False):
    size = 0
    if x > 0:
        size = int(np.log10(x)) + 1
    elif x < 0:
        size = int(np.log10(-x)) + 1
        if cminus:
            size += 1
    return size


def arg2dict(args, includes=None, excludes=None):
    kwargs = {}
    for arg in vars(args):
        if (includes is None or arg in includes) and (excludes is None or arg not in excludes):
            kwargs[arg] = getattr(args, arg)
    return kwargs


def print_args(args, keycolor='green', valuecolor='red'):
    if keycolor is None:
        keycolor = lambda x:x
    else:
        keycolor = eval('colors.{}'.format(keycolor))
    if valuecolor is None:
        valuecolor = lambda x:x
    else:
        valuecolor = eval('colors.{}'.format(valuecolor))
    for arg in vars(args):
        print('{}: {}'.format(keycolor(arg), valuecolor(getattr(args, arg))))


def dict2str(x, includes=None, excludes=None):
    assert isinstance(x, dict)
    string=None
    for k,v in x.items():
        if (includes is None or k in includes) and (excludes is None or k not in excludes):
            if string is None:
                string = '{}:{}'.format(k,v)
            else:
                string = '{}|{}:{}'.format(string, k, v)
    return string


def timestamp(date=None, fmt='%Y-%m-%d %H:%M:%S:%f', split='-'):
    """ time stamp function
        Attributes
        ==========
            date : string / None
                   if date is None, return current time stamp as string
            fmt : string
                  datetime format to parse `date` or to format current time
                  see https://docs.python.org/3.6/library/datetime.html/strftime-strptime-behavior
                  for more description on format
            split : char / string
                    char or string to replace whitespace and ':' in datetime string
                    For example
                    > split = '-'
                    > now = 2018-03-29 12:00:00
                    > output: 2018-03-29-12-00-00

        Returns
        ==========
            datetime if date is not None, otherwise current timestamp as string

        Raises:
        ==========
            ValueError if date is not None and fmt incomplete or ambiguous
    """
    if date is None:
        now = datetime.now()
        if split is not None:
            fmt = fmt.replace(' ', split).replace(':', split)
        fmt = '{{0:{}}}'.format(fmt)
        return fmt.format(now)
    else:
        return datetime.strptime(date, fmt)


def set_term_title(title):
    sys.stdout.write('\x1b]2;{}\x07'.format(title))


def set_seed(seed=1024):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
