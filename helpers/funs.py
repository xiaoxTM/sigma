from timeit import default_timer as timer
from .. import colors
import numpy as np
import math


def intsize(x, cminus=False):
    if x > 0:
        return int(np.log10(x)) + 1
    elif x == 0:
        return 1
    else:
        if cminus:
            return int(np.log10(-x)) + 2
        else:
            return int(np.log10(-x)) + 1


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



def line(iterable,
         epochs,
         iterations=None,
         brief=True,
         nprompts=10,
         feedbacks=False,
         timeit=False,
         show_total_time=False,
         accuracy=5,
         message=None,
         nc='x',
         cc='+'):
    """ Python Generator Tips [next|send / yield]
        show line message in a programmatic interactive mode
        to interactivate with line, call generator in the following order:
        (global_idx, elem, epoch, iteration) = (generator)
        while True:
            # do some work here
            (global_idx, elem, epoch, iteration) = generator.send(value)
            if need_break:
                break
        That is, to interactivate next / send with yield in generator, do
        it in the following loop:
                next(generator) \
                                 V
                    -------------|
                    |            |
                    |            V
                  send         yield
                    ^            |
                    |            |
                    |____________|

        Attibutes
        =========
            iterable : list / tuple / np.ndarray / range / None
                       iterable object or function that can yield something
                       generally speaking, iterable have length of:
                       "iterations * epochs"
            epochs : int or None
                     total iterations
                     if None, will be assigned as 1
            iterations : int / None
                         iterations per each epoch
                         NOTE:
                             - iterable is None, it will be assigned as
                               iterable = range(iterations * epochs)
                               therefore, iterations can not be None
                             - iterable is not None
                               - iterations is None, it will be assigned as
                                 iterations = len(iterable) / epochs
            brief : bool
                    if true, show extra information in form of
                        `[current-iter / total-iter]`
            nprompts : int
                       how many prompt (`nc`/`cc`) to mimic the progress
            feedbacks : boolean
                        if true, yield data and waiting for feedbacks from caller
                        else only yield data
            timeit : bool
                     time each iteration or not
            show_total_time : bool
                              show the sum of each iteration time if true
            accuracy : int
                       specifying print format precision for time
            message : string or None
                      message to be shown at the begining of each line
            nc : char
                 character to show as running prompt
            cc : char
                 character to show as running and ran prompt for each iteration
        Returns
        ==========
            tuple of (global_index, iterable_element, epoch, iteration), where:
            global_index : int
                            global index for iterable
            epoch : int
            iteration : int
                        iteration inside epoch
    """
    if epochs is None:
        epochs = 1
    if iterable is None:
        if iterations is None:
            raise ValueError('`iterable` and `iterations` can not both be None')
        iterable_size = iterations * epochs
        iterable = range(iterable_size)
    else:
        try:
            iterable_size = len(iterable)
        except TypeError:
            raise TypeError('getting the length of `iterable` failed')
        # number of iterations per each epoch
        if iterations is None:
            iterations = int(iterable_size / epochs)
    epochsize = intsize(epochs)
    if message is None:
        message = '@'
    nc = nc.encode('utf8')
    cc = cc.encode('utf8')
    # rotation times per one `+` / `x`
    scale = nprompts / iterations
    step = max(1, int(scale))
    _prompt = np.asarray([nc] * nprompts, dtype=np.string_)
    itersize = intsize(iterations)
    beg = None
    elapsed = None
    if brief:
        if timeit:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:3}}%]' \
                   ' {{}} @{{:0>.{}f}}(s){}'.format(epochsize,
                                                   message,
                                                   epochs,
                                                   nprompts,
                                                   accuracy,
                                                   colors.red('$'))
        else:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:3}}%] {{}}{}' \
                   .format(epochsize,
                           message,
                           epochs,
                           nprompts,
                           colors.red('$'))
    else:
        if timeit:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:{}}} / {{:{}}}, {{:3}}%]' \
                   ' {{}} @{{:0>.{}f}}(s){}'.format(epochsize,
                                                    message,
                                                    epochs,
                                                    nprompts,
                                                    itersize,
                                                    itersize,
                                                    accuracy,
                                                    colors.red('$'))
        else:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:{}}} / {{:{}}}, {{:3}}%]' \
                   ' {{}}{}'.format(epochsize,
                                    message,
                                    epochs,
                                    nprompts,
                                    itersize,
                                    itersize,
                                    colors.red('$'))
    def _line():
        totaltime = 0
        prev = 0
        idx = 0
        while True:
            elem = iterable[idx % iterable_size]
            time_beg = timer()
            epoch = int(idx / iterations)
            iteration = (idx % iterations)
            if feedbacks:
                ret = (yield (idx, elem, epoch, iteration))
            else:
                yield (idx, elem, epoch, iteration)
                ret = ''
            # convert iteration index to nprompts index
            block_beg = int(iteration * scale)
            if block_beg >= nprompts:
                block_beg = nprompts - step
            block_end = block_beg + step
            elapsed = round((timer() - time_beg), accuracy)
            totaltime += elapsed
            iteration += 1

            percentage = int(iteration * 100 / iterations)

            if _prompt[block_beg] == nc or prev != block_beg or percentage == 100:
                _prompt[block_beg:block_end] = cc
                if prev != block_beg:
                    _prompt[prev:block_beg] = cc
            else:
                _prompt[block_beg:block_end] = nc
            prev = block_beg
            prompt = _prompt[:block_end].tostring().decode('utf-8')
            if brief:
                if timeit:
                    message = spec.format(epoch+1, prompt, percentage,
                                          ret, elapsed)
                else:
                    message = spec.format(epoch+1, prompt, percentage, ret)
            else:
                if timeit:
                    message = spec.format(epoch+1, prompt, iteration, iterations,
                                          percentage, ret, elapsed)
                else:
                    message = spec.format(epoch+1, prompt, iteration, iterations,
                                          percentage, ret)
            end_flag = ''
            if percentage == 100:
                # start new epoch
                end_flag = '\n'
            print(message, end=end_flag, flush=True)
            idx += 1
        if show_total_time and timeit:
            print('\nTotal time elapsed:{}(s)'.format(totaltime))
        else:
            print()
    return _line, iterable_size
