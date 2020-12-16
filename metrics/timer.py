from contextlib import contextmanager
from timeit import default_timer as dtimer
from sigma.fontstyles import colors


def timeit(print_it):
    def _timeit(fun):
        def _wrap(*args, **kwargs):
            beg = dtimer()
            ret = fun(*args, **kwargs)
            end = dtimer()
            if print_it:
                print('Time elapsed for function {}: {}(s)'
                      .format(colors.blue(fun.__name__),
                              colors.red(end-beg)))
            return ret
        return _wrap
    return _timeit


@contextmanager
def time(message=None, color='blue'):
    if message is None:
        message = ''
    beg = dtimer()
    yield beg
    end = dtimer()
    print('>>> {}{}  <<<'.format(message, eval('colors.{}(end-bed)'.format(color))))