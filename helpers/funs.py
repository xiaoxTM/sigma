from timeit import default_timer as timer
from .. import colors

def timeit(print_it):
    def _timeit(fun):
        def _wrap(*args, **kwargs):
            beg = timer()
            ret = fun(*args, **kwargs)
            end = timer()
            if print_it:
                print('Time elapsed for function {}: {}(s)'
                      .format(colors.blue(fun.__name__),
                              colors.red(end-beg)))
            return ret
        return _wrap
    return _timeit
